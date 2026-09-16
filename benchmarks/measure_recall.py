"""
measure_recall.py — baseline(Docling HybridChunker) vs EvidenceChunker 벤치마크

같은 PDF+질문셋에서 두 코퍼스를 만들어 Recall@1 / EM을 비교한다.

    baseline : HybridChunker().chunk(doc) 전체 청크 (표 청크 포함, 원래 방식)
    EU       : EvidenceChunker.build_corpus() — EvidenceUnit + 비표 본문(카니발라이제이션 제거 후)
'
지표:
    Recall@1 (strict) : top-1이 EU 유닛이고 페이지가 정답과 일치
    Recall@1 (page)   : top-1 페이지만 일치 (hybrid 청크여도 인정)
    EM                : top-1 청크 텍스트에 answer_spec이 만족되는가

EM이 핵심 지표다. context_dependent는 표와 문단이 같은 페이지라 페이지 기준
Recall로는 변별이 안 되고, answer_spec의 비대칭 설계(value=표에만,
context_keys=문단에만) 때문에 EU만 구조적으로 통과 가능하다.

주의: normalize_for_em/has_token/em_hit은 benchmarks/generate_qa_docling.py의
동일 함수와 반드시 같은 규칙을 유지해야 한다 — 어긋나면 answer_spec이 의미를 잃는다.

통계: _ci()는 참고용 단순 근사치. 유의성 판단은 paired_statistics()의 문서 단위
클러스터 부트스트랩 CI(주 지표) + McNemar 검정(보조 지표, 질문 간 독립 가정)을 쓴다.

사용법:
    python measure_recall.py --pdf-dir ./data/pdfs --qa-dir ./auto_qa --out-dir ./results
    python measure_recall.py --pdf-dir ./data/pdfs --qa-dir ./auto_qa --out-dir ./results --dev-only
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 하이퍼파라미터 — 라이브러리 기본값과 동일하게 유지 (sweep은 별도 실험)
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENCODE_BATCH = 256
BBOX_THRESHOLD = 300.0
SIM_THRESHOLD = 0.00
CTX_WINDOW_PT = 300.0  # dist_pt 슬라이스 기준 (게이트 아님, BBOX_THRESHOLD와 같은 값)

HEADLINE_TYPES = {"cell_value", "table_about", "context_dependent"}
VALID_SUBSETS = {None, "main"}

# dev 서브셋 — 수동 QA셋 20개와 동일 문서 구성
DEV_DOCS = [
    "1. Attention is all you need", "6. DPO", "14. CLIP", "21. T5", "26. MMLU",
    "35. APB", "39. risk sharing", "42. rural housing", "45. gao-25-107649",
    "47. gao-26-107681", "50. gao-26-107884", "52. gao-26-108011",
    "55. gao-26-108116", "60. ieee1", "64. ieee5", "66. ieee7", "70. ieee11",
    "72. ieee13", "80. ssrn-1331573", "85. ssrn-2760631",
]

# 문서 매크로 평균에서 이 문항 수 미만은 제외한다. QA셋은 문서당 1~122문항으로
# 편차가 커서(median 20), 문항 적은 문서가 매크로 평균에서 과대 대표되는 걸 막는다.
MIN_DOC_N = 10


# ===========================================================================
# 1. 채점 — generate_qa_docling.py §1과 동일 규칙
# ===========================================================================

def normalize_for_em(s: str) -> str:
    if not s:
        return ""
    s = s.lower().replace("\xa0", " ")
    s = re.sub(r"[^\w.\-±× ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\.$", "", s)
    return s


def has_token(needle: str, haystack_norm: str) -> bool:
    n = normalize_for_em(needle)
    if not n:
        return False
    return re.search(r"(?<!\d)" + re.escape(n) + r"(?!\d)", haystack_norm) is not None


def em_hit(spec: dict, chunk_text: str) -> bool:
    """값 + 모든 context_keys가 같은 청크 안에 있으면 정답.

    EU의 'A | B: v'도 baseline의 '| A | v |'도 동일하게 판정된다 — EU 전용
    통과 경로 없음(생성 단계 CI 불변식으로 전수 검증됨).
    """
    if not spec or not chunk_text:
        return False
    c = normalize_for_em(chunk_text)
    if not has_token(spec.get("value", ""), c):
        return False
    return all(has_token(k, c) for k in spec.get("context_keys", []))


def _legacy_answer_in_chunk(answer: str, chunk_text: str) -> bool:
    """answer_spec 없는 구버전 수동 QA셋 호환용 폴백."""
    a, c = normalize_for_em(answer), normalize_for_em(chunk_text)
    if not a or not c:
        return False
    if re.search(r"(?<!\d)" + re.escape(a) + r"(?!\d)", c):
        return True
    tokens = [t for t in a.split() if len(t) >= 4]
    if not tokens:
        return False
    return sum(1 for t in tokens if has_token(t, c)) / len(tokens) >= 0.8


def score_em(qa: dict, chunk_text: str) -> bool:
    spec = qa.get("answer_spec")
    return em_hit(spec, chunk_text) if spec else _legacy_answer_in_chunk(qa.get("answer", ""), chunk_text)


def classify_real_driver(b_ok: bool, e_ok: bool) -> str:
    if b_ok and e_ok:
        return "both_right"
    if b_ok and not e_ok:
        return "baseline_win_eu_lose"
    if not b_ok and e_ok:
        return "eu_win_baseline_lose"
    return "both_wrong"


# ===========================================================================
# 2. PDF ↔ QA 매칭
# ===========================================================================

_LEADING_NUM = re.compile(r"^(\d+)\.")


def _sort_key(p: Path):
    m = _LEADING_NUM.match(p.name)
    return (int(m.group(1)) if m else 10 ** 6, p.name)


def pdf_qa_pairs(pdf_dir: Path, qa_dir: Path, dev_only: bool, max_pdfs: int | None) -> list[tuple[Path, Path, str]]:
    """(pdf_path, qa_path, doc_id) 목록. auto_qa는 파일명이 PDF와 1:1 대응한다."""
    pairs, unmatched, empty = [], [], []
    for qa_path in sorted(qa_dir.glob("*_qa.json"), key=_sort_key):
        if qa_path.name.startswith("_"):
            continue
        doc_id = qa_path.stem[:-3] if qa_path.stem.endswith("_qa") else qa_path.stem
        if dev_only and doc_id not in DEV_DOCS:
            continue
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            unmatched.append(qa_path.name)
            continue
        try:
            n = len(json.load(open(qa_path, encoding="utf-8")))
        except Exception:
            n = 0
        if n == 0:
            empty.append(qa_path.name)
            continue
        pairs.append((pdf_path, qa_path, doc_id))

    if unmatched:
        print(f"  [warn] PDF 매칭 실패 {len(unmatched)}건: {unmatched[:5]}")
    if empty:
        print(f"  [skip] 0문항 파일 {len(empty)}건: {empty}")
    if dev_only:
        missing = [d for d in DEV_DOCS if d not in {p[2] for p in pairs}]
        if missing:
            print(f"  [warn] dev 목록에 있으나 못 찾음: {missing}")
    print(f"  [pairs] {len(pairs)} PDF+QA")
    return pairs[:max_pdfs] if max_pdfs else pairs


def chunk_page(c) -> "int | None":
    """HybridChunker 청크의 페이지 번호."""
    try:
        for di in c.meta.doc_items:
            if di.prov:
                return di.prov[0].page_no
    except Exception:
        pass
    return None


# ===========================================================================
# 3. 코퍼스 구성 — PDF 1회 파싱으로 baseline·EU 동시 생성
# ===========================================================================
# EvidenceChunker.build_corpus()를 그대로 쓰면 Docling 변환이 2회(baseline용 +
# EU용) 일어난다. 아래는 build_corpus()와 동일한 로직을 doc 재사용 형태로
# 편 것이며 라이브러리 함수만 호출한다(재구현 아님). PARITY_CHECK=True로
# 첫 문서에서 build_corpus() 결과와 대조해 동등성을 확인할 수 있다.

_converter = None


def get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter
        _converter = DocumentConverter()
    return _converter


def build_both_corpora(pdf_path: Path, doc_id: str, parity_check: bool = False):
    """Returns (baseline: list[HybridChunker chunk], eu_corpus: list[RetrievalChunk], stats: dict)."""
    from docling.chunking import HybridChunker
    from docling_core.types.doc import DocItemLabel
    from evidence_chunker.chunker import build_evidence_units
    from evidence_chunker.split import split_oversized_units
    from evidence_chunker.parser.docling import DoclingParser
    from evidence_chunker.export import TextChunk, filter_consumed_paragraphs

    doc = get_converter().convert(str(pdf_path)).document

    all_chunks = list(HybridChunker().chunk(doc))
    is_table_chunk = lambda c: any(di.label == DocItemLabel.TABLE for di in c.meta.doc_items)

    parsed = DoclingParser().from_doc(doc)
    eu_list = build_evidence_units(parsed, BBOX_THRESHOLD, SIM_THRESHOLD, doc_id)
    eu_list = split_oversized_units(eu_list)

    non_table = [c for c in all_chunks if not is_table_chunk(c)]
    before_dedup = len(non_table)
    non_table = filter_consumed_paragraphs(non_table, eu_list)
    eu_corpus = eu_list + [TextChunk(c, doc_id, i) for i, c in enumerate(non_table)]

    conf = Counter(eu.caption_confidence for eu in eu_list)
    stats = {
        "n_tables": len(parsed.tables),
        "n_eu": len(eu_list),
        "n_split": sum(1 for eu in eu_list if eu.is_split),
        "n_baseline_chunks": len(all_chunks),
        "n_hybrid_before_dedup": before_dedup,
        "n_consumed_removed": before_dedup - len(non_table),
        "n_poisoned_captions": sum(1 for eu in eu_list if eu.caption_text and not eu.safe_caption),
        "caption_confidence": dict(conf),
    }

    if parity_check:
        from evidence_chunker import EvidenceChunker
        ref = EvidenceChunker(bbox_threshold=BBOX_THRESHOLD,
                               sim_threshold=SIM_THRESHOLD).build_corpus(pdf_path, doc_id=doc_id)
        a = [c.chunk_id for c in eu_corpus]
        b = [c.chunk_id for c in ref]
        print(f"  [parity] build_corpus()와 동일: {a == b}  ({len(a)} vs {len(b)})")

    del doc
    gc.collect()
    return all_chunks, eu_corpus, stats


# ===========================================================================
# 4. 문서 1개 평가
# ===========================================================================

def evaluate_one(pdf_path: Path, qa_path: Path, doc_id: str, model, device: str, parity_check: bool = False) -> dict:
    import numpy as np

    print(f"\n{'-'*62}")
    print(f"  {doc_id}")

    try:
        b_chunks, eu_corpus, stats = build_both_corpora(pdf_path, doc_id, parity_check)
    except Exception as e:
        print(f"  [ERR] {type(e).__name__}: {e}")
        return {}

    print(f"  표 {stats['n_tables']} → EU {stats['n_eu']} (분할 {stats['n_split']})  |  "
          f"baseline 청크 {stats['n_baseline_chunks']}")
    print(f"  카니발라이제이션 제거 {stats['n_consumed_removed']}/{stats['n_hybrid_before_dedup']}  |  "
          f"caption {stats['caption_confidence']}")

    b_texts = [c.text for c in b_chunks]
    b_pages = [chunk_page(c) for c in b_chunks]

    e_units, e_src, e_pages, e_disp = [], [], [], []
    for c in eu_corpus:
        page = c.metadata.get("page_no")
        for u in c.retrieval_units:
            e_units.append(u)
            e_src.append(c.chunk_id)
            e_pages.append(page)
            e_disp.append(c.text)

    if not b_texts or not e_units:
        print("  [ERR] 빈 코퍼스")
        return {}
    print(f"  EU 코퍼스 {len(e_units)} 유닛  (EU {stats['n_eu']} + hybrid "
          f"{len(eu_corpus) - stats['n_eu']})")

    enc = lambda xs: model.encode(xs, normalize_embeddings=True,
                                   show_progress_bar=False, batch_size=ENCODE_BATCH)
    b_emb, e_emb = enc(b_texts), enc(e_units)

    qa_list = json.load(open(qa_path, encoding="utf-8"))
    usable, skipped = [], 0
    for qa in qa_list:
        if qa.get("question_delabeled") is None or qa.get("subset", "main") not in VALID_SUBSETS:
            skipped += 1
            continue
        usable.append(qa)
    if not usable:
        print("  [warn] 사용 가능 문항 0")
        return {}

    q_emb = enc([q["question_delabeled"] for q in usable])
    b_top = np.dot(q_emb, b_emb.T).argmax(axis=1)
    e_top = np.dot(q_emb, e_emb.T).argmax(axis=1)

    rows = []
    for i, qa in enumerate(usable):
        exp = qa.get("page")
        meta = qa.get("meta") or {}

        bi = int(b_top[i])
        b_ok = b_pages[bi] is not None and b_pages[bi] == exp
        b_em = score_em(qa, b_texts[bi])

        ei = int(e_top[i])
        src, epage = e_src[ei], e_pages[ei]
        is_eu_unit = "-hybrid-" not in src  # TextChunk는 "{doc_id}-hybrid-{i}"
        e_ok_strict = is_eu_unit and epage == exp
        e_ok_page = epage == exp
        e_em = score_em(qa, e_disp[ei])

        if e_ok_strict:
            fail = None
        elif not is_eu_unit:
            fail = "hybrid won" + (" (page O)" if e_ok_page else " (page X)")
        else:
            fail = f"EU wrong page (p{epage}, exp p{exp})"

        rows.append({
            "doc_id": doc_id, "qid": qa.get("qid", ""), "type": qa.get("type", "unknown"),
            "question": qa["question_delabeled"], "expected_page": exp,
            "b_correct": b_ok, "b_em": b_em,
            "e_source": src, "e_page": epage,
            "e_correct": e_ok_strict, "e_page_correct": e_ok_page, "e_em": e_em,
            "fail_reason": fail,
            "real_driver": classify_real_driver(b_ok, e_ok_strict),
            "e_top1_text": e_units[ei][:120],
            "dist_pt": meta.get("dist_pt"),
            "ctx_explicit_ref": meta.get("ctx_explicit_ref"),
            "n_tables_on_page": meta.get("n_tables_on_page"),
            "page_index": meta.get("page_index"),
            "header_rows": meta.get("header_rows"),
        })

    n = len(rows)
    print(f"\n  [{n}문항, 제외 {skipped}]")
    print(f"    baseline  R {sum(r['b_correct'] for r in rows)/n:.3f}   EM {sum(r['b_em'] for r in rows)/n:.3f}")
    print(f"    EU        R {sum(r['e_correct'] for r in rows)/n:.3f}   EM {sum(r['e_em'] for r in rows)/n:.3f}")
    for t in sorted({r["type"] for r in rows}):
        s = [r for r in rows if r["type"] == t]
        print(f"      {t:20s} n={len(s):4d}  "
              f"R {sum(r['b_correct'] for r in s)/len(s):.3f}->{sum(r['e_correct'] for r in s)/len(s):.3f}  "
              f"EM {sum(r['b_em'] for r in s)/len(s):.3f}->{sum(r['e_em'] for r in s)/len(s):.3f}")

    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
    return {"doc_id": doc_id, "rows": rows, "skipped": skipped, **stats}


# ===========================================================================
# 5. 집계 / 리포트
# ===========================================================================

def _rate(rows, key):
    return (sum(r[key] for r in rows) / len(rows)) if rows else None


def _line(label, rows, width=28):
    if not rows:
        return f"  {label:<{width}} n=    0        -                    -"
    n = len(rows)
    bR, eR = _rate(rows, "b_correct"), _rate(rows, "e_correct")
    bE, eE = _rate(rows, "b_em"), _rate(rows, "e_em")
    return (f"  {label:<{width}} n={n:5d}   "
            f"R {bR:.3f}->{eR:.3f} ({(eR-bR)*100:+5.1f}pp)   "
            f"EM {bE:.3f}->{eE:.3f} ({(eE-bE)*100:+5.1f}pp)")


def _ci(n):
    """95% CI 반폭(pp) — 표본이 작을 때 관측된 갭이 실제로 유의한지 판단하는 기준.

    단순 이항분포 최대분산(p=0.5) 근사(Wald)라, 같은 질문에 대한
    baseline/EU 쌍대비교 구조나 문서 내 상관은 반영하지 않는다. 엄밀한
    유의성 판단에는 아래 paired_statistics()를 쓸 것 — 이 함수는 빠른
    참고용 오차범위로만 남겨둔다.
    """
    return 1.96 * 0.5 / (n ** 0.5) * 100 if n else float("inf")


def paired_statistics(doc_ids, baseline, treatment, repeats: int = 50000, seed: int = 20260915) -> dict:
    """문서 단위 클러스터 부트스트랩 CI + 보조 McNemar 검정.

    _ci()와 달리 같은 문서에서 나온 질문들을 묶어서(문서를 리샘플링 단위로
    삼아) 차이의 95% percentile CI를 구한다 — 문서 내 질문 간 상관을
    반영하는 방식. McNemar는 "baseline만 맞음 vs EU만 맞음"의 비대칭을
    검정하는 보조 지표로 덧붙이되, 문서 간 의존성은 보정하지 않는다는
    가정을 명시한다.

    baseline/treatment: 질문별 0/1(또는 bool) 정오답 배열. doc_ids와 길이가
    같아야 하며, 같은 인덱스가 같은 질문을 가리켜야 한다(쌍대비교 전제).
    """
    b = np.asarray(list(baseline), dtype=np.int64)
    e = np.asarray(list(treatment), dtype=np.int64)
    doc_ids = list(doc_ids)
    if len(doc_ids) != len(b) or len(b) != len(e) or not len(b):
        raise ValueError("Paired vectors must be nonempty and equal length.")

    grouped: dict = defaultdict(lambda: [0, 0])  # doc_id -> [n_questions, sum(e-b)]
    for d, delta in zip(doc_ids, e - b):
        grouped[d][0] += 1
        grouped[d][1] += int(delta)
    a = np.asarray([grouped[d] for d in sorted(grouped)], dtype=np.int64)

    ci = None
    if len(a) >= 2:
        rng, values = np.random.default_rng(seed), []
        for start in range(0, repeats, 2048):
            idx = rng.integers(0, len(a), size=(min(2048, repeats - start), len(a)))
            total = a[idx].sum(axis=1)
            values.append(100 * total[:, 1] / total[:, 0])
        ci = np.quantile(np.concatenate(values), [.025, .975]).tolist()

    b_only = int(((b == 1) & (e == 0)).sum())
    e_only = int(((b == 0) & (e == 1)).sum())
    discordant = b_only + e_only
    chi2 = max(abs(b_only - e_only) - 1, 0) ** 2 / discordant if discordant else 0.0
    p = math.erfc(math.sqrt(chi2 / 2)) if discordant else 1.0

    return {
        "n_questions": len(b), "n_documents": len(a),
        "baseline": float(b.mean()), "treatment": float(e.mean()),
        "difference_pp": float(100 * (e - b).mean()),
        "cluster_bootstrap_ci95_pp": ci,
        "bootstrap": {"unit": "document", "statistic": "micro_rate_difference",
                      "method": "percentile", "repeats": repeats, "seed": seed},
        "mcnemar_supplementary": {
            "method": "chi_square_continuity_corrected", "chi2": chi2, "p_value": p,
            "baseline_only": b_only, "treatment_only": e_only,
            "assumption": "질문 간 독립 가정 — 문서 내 의존성은 보정하지 않음(보조 지표)",
        },
        "ci_note": ("문서를 독립 표집 단위로 취급한 근사치이며, LLM 생성 답변 정확도의 CI가 아님"
                    if ci is not None else "문서 2개 미만이라 CI 계산 불가"),
    }


def run(pdf_dir: Path, qa_dir: Path, out_dir: Path, dev_only: bool, max_pdfs: int | None,
        parity_check: bool = False, use_mlflow: bool = False) -> None:
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "dev20" if dev_only else "full90"

    if use_mlflow:
        import mlflow
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow_db = (Path.cwd() / "mlflow.db").resolve()
            mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
        mlflow.set_experiment("evidence-chunker-benchmark")
        mlflow.start_run(run_name=tag)
        mlflow.log_params({
            "tag": tag,
            "bbox_threshold": BBOX_THRESHOLD,
            "sim_threshold": SIM_THRESHOLD,
            "embed_model": EMBED_MODEL_NAME,
            "encode_batch": ENCODE_BATCH,
            "min_doc_n": MIN_DOC_N,
            "max_pdfs": max_pdfs,
        })

    pairs = pdf_qa_pairs(pdf_dir, qa_dir, dev_only, max_pdfs)
    if not pairs:
        print("[ERR] PDF-QA 쌍 없음")
        if use_mlflow:
            mlflow.end_run(status="FAILED")
        return

    import evidence_chunker
    print(f"[setup] evidence_chunker {evidence_chunker.__version__}  device={device}")
    print(f"[setup] scope={'dev 20' if dev_only else 'full 90'}")
    print(f"[setup] bbox={BBOX_THRESHOLD}pt  sim={SIM_THRESHOLD}  batch={ENCODE_BATCH}")
    print(f"\n[model] {EMBED_MODEL_NAME} on {device}")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    results, rows = [], []
    for i, (pdf, qa, doc_id) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}]", end="")
        r = evaluate_one(pdf, qa, doc_id, model, device, parity_check)
        if r:
            results.append(r)
            rows += r["rows"]

    if not rows:
        print("[ERR] 결과 없음")
        if use_mlflow:
            mlflow.end_run(status="FAILED")
        return

    N = len(rows)
    print(f"\n{'='*100}")
    print(f"  HEADLINE — {len(results)} PDF, {N} 문항  ({'dev 20' if dev_only else 'full 90'})")
    print(f"{'='*100}")
    print(_line("전체", rows))

    print(f"\n  ── 유형별 ──")
    for t in ("cell_value", "table_about", "context_dependent"):
        print(_line(t, [r for r in rows if r["type"] == t]))

    ctx = [r for r in rows if r["type"] == "context_dependent"]
    if ctx:
        print(f"\n  ── context_dependent 상세 ──")
        inw = [r for r in ctx if r["dist_pt"] is not None and r["dist_pt"] <= CTX_WINDOW_PT]
        outw = [r for r in ctx if r["dist_pt"] is not None and r["dist_pt"] > CTX_WINDOW_PT]
        print(_line(f"dist <= {CTX_WINDOW_PT:.0f}pt (창 안)", inw))
        print(_line(f"dist >  {CTX_WINDOW_PT:.0f}pt (창 밖)", outw))
        if outw:
            gap = (_rate(outw, "e_em") - _rate(outw, "b_em")) * 100
            half = _ci(len(outw))
            print(f"     창 밖 n={len(outw)}  EM 갭 {gap:+.1f}pp  CI ±{half:.1f}pp")
        print(_line("explicit_ref=True", [r for r in ctx if r["ctx_explicit_ref"] is True]))
        print(_line("explicit_ref=False", [r for r in ctx if r["ctx_explicit_ref"] is False]))

    print(f"\n  ── meta 슬라이스 ──")
    print(_line("cross-table (표>=2/page)", [r for r in rows if (r["n_tables_on_page"] or 0) >= 2]))
    print(_line("single-table (표=1/page)", [r for r in rows if (r["n_tables_on_page"] or 0) == 1]))
    print(_line("ToC 구간 (page_index<=.1)",
                [r for r in rows if r["page_index"] is not None and r["page_index"] <= 0.1]))
    print(_line("다단 헤더 (header_rows>1)", [r for r in rows if (r["header_rows"] or 1) > 1]))

    # 문서 매크로 평균 — 문항 수 편차(1~122문항)로 인한 왜곡을 micro 지표와 함께 교차 확인
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)

    def _macro(docs):
        if not docs:
            return None
        n = len(docs)
        return {k: sum(_rate(v, k) for v in docs) / n
                for k in ("b_correct", "e_correct", "b_em", "e_em")}

    all_docs = list(by_doc.values())
    big_docs = [v for v in all_docs if len(v) >= MIN_DOC_N]
    ma, mbig = _macro(all_docs), _macro(big_docs)

    def _mline(label, m, n):
        if not m:
            return f"  {label:<28} n={n:5d}   -"
        return (f"  {label:<28} n={n:5d}   "
                f"R {m['b_correct']:.3f}->{m['e_correct']:.3f} "
                f"({(m['e_correct']-m['b_correct'])*100:+5.1f}pp)   "
                f"EM {m['b_em']:.3f}->{m['e_em']:.3f} "
                f"({(m['e_em']-m['b_em'])*100:+5.1f}pp)")

    print(f"\n  ── 문서 매크로 평균 ──")
    print(_mline("전체 문서", ma, len(all_docs)))
    print(_mline(f"{MIN_DOC_N}문항 이상만", mbig, len(big_docs)))
    mb, me = ma["b_correct"], ma["e_correct"]
    mbe, mee = ma["b_em"], ma["e_em"]

    rd = Counter(r["real_driver"] for r in rows)
    print(f"\n  [real_driver] both_right={rd['both_right']}  "
          f"baseline_win_eu_lose={rd['baseline_win_eu_lose']}  "
          f"eu_win_baseline_lose={rd['eu_win_baseline_lose']}  both_wrong={rd['both_wrong']}")

    paired_em = paired_statistics([r["doc_id"] for r in rows],
                                   [r["b_em"] for r in rows], [r["e_em"] for r in rows])
    ci = paired_em["cluster_bootstrap_ci95_pp"]
    ci_str = f"[{ci[0]:+.1f}, {ci[1]:+.1f}]pp" if ci else "N/A(문서<2)"
    mc = paired_em["mcnemar_supplementary"]
    print(f"\n  [EM 통계 검정] 문서 단위 부트스트랩 차이 {paired_em['difference_pp']:+.1f}pp  "
          f"95% CI {ci_str}")
    print(f"  [McNemar 보조] baseline만 정답 {mc['baseline_only']}  EU만 정답 {mc['treatment_only']}  "
          f"chi2={mc['chi2']:.2f}  p={mc['p_value']:.2e}")

    fails = Counter(r["fail_reason"] for r in rows if r["fail_reason"])
    if fails:
        print(f"\n  EU 실패 사유 상위")
        for k, v in fails.most_common(8):
            print(f"    {v:5d}  {k}")

    tc = sum(r["n_consumed_removed"] for r in results)
    tb = sum(r["n_hybrid_before_dedup"] for r in results)
    tp = sum(r["n_poisoned_captions"] for r in results)
    ts = sum(r["n_split"] for r in results)
    conf = Counter()
    for r in results:
        conf.update(r["caption_confidence"])
    ca = sum(conf.values()) or 1
    print(f"\n  [파이프라인] EU {sum(r['n_eu'] for r in results)} (분할 {ts})  "
          f"dedup {tc}/{tb}  figure-caption 필터 {tp}")
    print(f"  [caption_confidence] direct {conf['direct']} ({conf['direct']/ca:.0%})  "
          f"inferred {conf['inferred']} ({conf['inferred']/ca:.0%})  "
          f"none {conf['none']} ({conf['none']/ca:.0%})")

    print(f"\n{'-'*100}")
    print(f"  {'문서':40s} {'N':>5} {'base_R':>7} {'EU_R':>7} {'base_EM':>8} {'EU_EM':>7}")
    print(f"{'-'*100}")
    for d, rs in sorted(by_doc.items(),
                         key=lambda kv: _rate(kv[1], "e_correct") - _rate(kv[1], "b_correct")):
        bR, eR = _rate(rs, "b_correct"), _rate(rs, "e_correct")
        flag = " *" if eR > bR else ("  " if abs(eR - bR) < 1e-9 else " v")
        print(f"  {d[:40]:40s} {len(rs):5d} {bR:7.3f} {eR:7.3f} "
              f"{_rate(rs,'b_em'):8.3f} {_rate(rs,'e_em'):7.3f}{flag}")

    def blk(rs):
        if not rs:
            return None
        return {"n": len(rs),
                "baseline_recall": round(_rate(rs, "b_correct"), 4),
                "eu_recall": round(_rate(rs, "e_correct"), 4),
                "baseline_em": round(_rate(rs, "b_em"), 4),
                "eu_em": round(_rate(rs, "e_em"), 4),
                "ci_halfwidth_pp": round(_ci(len(rs)), 2)}

    summary = {
        "config": {"scope": tag, "qa_dir": str(qa_dir), "embed_model": EMBED_MODEL_NAME,
                   "bbox_threshold": BBOX_THRESHOLD, "sim_threshold": SIM_THRESHOLD,
                   "scoring": "answer_spec / em_hit", "evidence_chunker": evidence_chunker.__version__},
        "overall": blk(rows),
        "by_type": {t: blk([r for r in rows if r["type"] == t])
                    for t in ("cell_value", "table_about", "context_dependent")},
        "context_dependent_slices": {
            "in_window": blk([r for r in ctx if r["dist_pt"] is not None and r["dist_pt"] <= CTX_WINDOW_PT]),
            "out_window": blk([r for r in ctx if r["dist_pt"] is not None and r["dist_pt"] > CTX_WINDOW_PT]),
            "explicit_ref": blk([r for r in ctx if r["ctx_explicit_ref"] is True]),
            "no_explicit_ref": blk([r for r in ctx if r["ctx_explicit_ref"] is False]),
        } if ctx else {},
        "meta_slices": {
            "cross_table": blk([r for r in rows if (r["n_tables_on_page"] or 0) >= 2]),
            "single_table": blk([r for r in rows if (r["n_tables_on_page"] or 0) == 1]),
            "toc_zone": blk([r for r in rows if r["page_index"] is not None and r["page_index"] <= 0.1]),
            "multi_header": blk([r for r in rows if (r["header_rows"] or 1) > 1]),
        },
        "macro_average": {"documents": len(all_docs),
                          "baseline_recall": round(mb, 4), "eu_recall": round(me, 4),
                          "baseline_em": round(mbe, 4), "eu_em": round(mee, 4)},
        "macro_average_min_n": ({"documents": len(big_docs), "min_questions": MIN_DOC_N,
                                 "baseline_recall": round(mbig["b_correct"], 4),
                                 "eu_recall": round(mbig["e_correct"], 4),
                                 "baseline_em": round(mbig["b_em"], 4),
                                 "eu_em": round(mbig["e_em"], 4)} if mbig else None),
        "doc_question_counts": {d: len(v) for d, v in by_doc.items()},
        "real_driver": dict(rd),
        "paired_statistics_em": paired_em,
        "fail_reasons": dict(fails),
        "pipeline": {"n_eu": sum(r["n_eu"] for r in results), "n_split": ts,
                     "dedup_removed": tc, "hybrid_before_dedup": tb,
                     "poisoned_captions": tp, "caption_confidence": dict(conf)},
        "by_doc": {d: blk(rs) for d, rs in by_doc.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"bench_{tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"bench_{tag}_rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: bench_{tag}.json  /  bench_{tag}_rows.json ({len(rows)} rows)")

    if use_mlflow:
        mlflow.log_metrics(summary["macro_average"])
        if summary.get("macro_average_min_n"):
            mlflow.log_metrics({
                f"minN_{k}": v for k, v in summary["macro_average_min_n"].items()
                if isinstance(v, (int, float))
            })
        # QA 유형별(cell_value / table_about / context_dependent) 지표도
        # DagsHub에서 바로 비교할 수 있게 별도 metric으로 남긴다.
        for t, blk_v in summary.get("by_type", {}).items():
            if not blk_v:
                continue
            mlflow.log_metrics({
                f"type_{t}_{k}": v for k, v in blk_v.items()
                if isinstance(v, (int, float))
            })
        mlflow.log_artifact(str(out_dir / f"bench_{tag}.json"))
        mlflow.log_artifact(str(out_dir / f"bench_{tag}_rows.json"))
        mlflow.end_run()


# ===========================================================================
# 6. CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--pdf-dir", type=Path, required=True, help="벤치마크 PDF 디렉토리")
    p.add_argument("--qa-dir", type=Path, required=True,
                    help="generate_qa_docling.py가 만든 {문서}_qa.json이 있는 디렉토리")
    p.add_argument("--out-dir", type=Path, required=True, help="결과 JSON을 저장할 디렉토리")
    p.add_argument("--dev-only", action="store_true", help="dev 서브셋(20개 문서)만 실행")
    p.add_argument("--max-pdfs", type=int, default=None, help="추가 상한 (디버깅용)")
    p.add_argument("--parity-check", action="store_true",
                    help="첫 문서에서 EvidenceChunker.build_corpus() 결과와 대조")
    p.add_argument("--mlflow", action="store_true")
    # 스윕 자동화
    p.add_argument("--bbox-threshold", type=float, default=None, help="BBOX_THRESHOLD")
    p.add_argument("--sim-threshold", type=float, default=None, help="SIM_THRESHOLD")
    p.add_argument("--embed-model", type=str, default=None, help="EMBED_MODEL_NAME")
    p.add_argument("--encode-batch", type=int, default=None, help="ENCODE_BATCH")
    return p.parse_args()


def main() -> None:
    global BBOX_THRESHOLD, SIM_THRESHOLD, EMBED_MODEL_NAME, ENCODE_BATCH, CTX_WINDOW_PT

    args = parse_args()

    if args.bbox_threshold is not None:
        BBOX_THRESHOLD = args.bbox_threshold
        CTX_WINDOW_PT = args.bbox_threshold  # 두 값은 항상 같이 움직임
    if args.sim_threshold is not None:
        SIM_THRESHOLD = args.sim_threshold
    if args.embed_model is not None:
        EMBED_MODEL_NAME = args.embed_model
    if args.encode_batch is not None:
        ENCODE_BATCH = args.encode_batch

    run(args.pdf_dir, args.qa_dir, args.out_dir, args.dev_only, args.max_pdfs, args.parity_check,
        args.mlflow)


if __name__ == "__main__":
    main()
