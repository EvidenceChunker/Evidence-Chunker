"""
benchmark.py — baseline(Docling HybridChunker) vs EvidenceChunker 벤치마크

유일한 평가 엔진. measure_recall.py는 이 파일의 run()을 호출하는 CI 전용
얇은 래퍼(DagsHub/MLflow 로깅만 담당)이므로, 채점 로직은 여기만 고치면 된다.

지표: PageHit@1/5/10(모든 arm 동일 기준), evidence_hit@1(정답 값+문맥 키
포함 여부), TableHit@k(정답 표 자체 일치, 페이지 일치보다 엄격), ablation
(caption/context 제거), 문서 단위 부트스트랩 CI + McNemar, 동일 토큰 예산
LLM 실답변 채점(Gemini).

사전 준비(직접 설치할 것 — 이 스크립트는 설치를 수행하지 않는다):
    pip install docling docling-core sentence-transformers langchain-core \\
                google-genai tiktoken
    pip install -e .

사용법:
    python benchmark.py --pdf-dir ./data/pdfs --qa-dir ./auto_qa \\
        --out-dir ./results --dev-only
    python benchmark.py --pdf-dir ./data/pdfs --qa-dir ./auto_qa \\
        --out-dir ./results --gemini-api-key $GEMINI_API_KEY --llm-sample-n 300

    GEMINI_API_KEY가 없으면 LLM 실답변 평가만 자동으로 건너뛴다.
    CI에서는 measure_recall.py를 통해 실행된다.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENCODE_BATCH = 256
BBOX_THRESHOLD = 300.0
SIM_THRESHOLD = 0.00
K_LIST = (1, 5, 10)
TOP_K_MAX = max(K_LIST)

CONTEXT_TOKEN_BUDGET = 1500
LLM_SAMPLE_N = 300
LLM_SAMPLE_SEED = 20260915
MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash",
]

# dev 서브셋 — v04와 동일 문서 구성
DEV_DOCS = [
    "1. Attention is all you need", "6. DPO", "14. CLIP", "21. T5", "26. MMLU",
    "35. APB", "39. risk sharing", "42. rural housing", "45. gao-25-107649",
    "47. gao-26-107681", "50. gao-26-107884", "52. gao-26-108011",
    "55. gao-26-108116", "60. ieee1", "64. ieee5", "66. ieee7", "70. ieee11",
    "72. ieee13", "80. ssrn-1331573", "85. ssrn-2760631",
]
_LEADING_NUM = re.compile(r"^(\d+)\.")


# ===========================================================================
# 1. 채점 함수
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


def evidence_hit(spec: Optional[dict], chunk_text: str) -> bool:
    """정답 값 + 모든 context_keys가 같은 청크 안에 있으면 True. 구 이름 em_hit."""
    if not spec or not chunk_text:
        return False
    c = normalize_for_em(chunk_text)
    if not has_token(spec.get("value", ""), c):
        return False
    return all(has_token(k, c) for k in spec.get("context_keys", []))


def classify_real_driver(b_ok: bool, e_ok: bool) -> str:
    if b_ok and e_ok:
        return "both_right"
    if b_ok and not e_ok:
        return "baseline_win_eu_lose"
    if not b_ok and e_ok:
        return "eu_win_baseline_lose"
    return "both_wrong"


def paired_statistics(doc_ids, baseline, treatment, repeats: int = 50000, seed: int = 20260915) -> dict:
    """문서 단위 클러스터 부트스트랩 CI(주 지표) + McNemar 검정(보조 지표)."""
    b = np.asarray(list(baseline), dtype=np.int64)
    e = np.asarray(list(treatment), dtype=np.int64)
    doc_ids = list(doc_ids)
    if len(doc_ids) != len(b) or len(b) != len(e) or not len(b):
        raise ValueError("Paired vectors must be nonempty and equal length.")

    grouped = defaultdict(lambda: [0, 0])
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


def _self_test() -> None:
    """Docling/GPU 없이 채점 로직만 검증. import 시 자동 실행."""
    spec = {"value": "42.3", "context_keys": ["APAC"]}
    assert evidence_hit(spec, "APAC region | Q2: 42.3") is True
    assert evidence_hit(spec, "42.3 percent, unrelated") is False
    assert evidence_hit(spec, "APAC region only, no numbers") is False

    assert classify_real_driver(True, True) == "both_right"
    assert classify_real_driver(True, False) == "baseline_win_eu_lose"
    assert classify_real_driver(False, True) == "eu_win_baseline_lose"
    assert classify_real_driver(False, False) == "both_wrong"

    doc_ids, b, e = [], [], []
    for _ in range(1181): doc_ids.append("d0"); b.append(1); e.append(1)
    for _ in range(521):  doc_ids.append("d0"); b.append(0); e.append(1)
    for _ in range(382):  doc_ids.append("d0"); b.append(1); e.append(0)
    for _ in range(641):  doc_ids.append("d0"); b.append(0); e.append(0)
    r = paired_statistics(doc_ids, b, e, repeats=2000)
    assert abs(r["mcnemar_supplementary"]["chi2"] - 21.0897) < 0.01, r
    assert abs(r["mcnemar_supplementary"]["p_value"] - 4.3828e-06) < 1e-8, r
    assert r["cluster_bootstrap_ci95_pp"] is None


_self_test()


# ===========================================================================
# 2. PDF <-> QA 매칭
# ===========================================================================

def _sort_key(p: Path):
    m = _LEADING_NUM.match(p.name)
    return (int(m.group(1)) if m else 10 ** 6, p.name)


def pdf_qa_pairs(pdf_dir: Path, qa_dir: Path, dev_only: bool, max_pdfs):
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
    print(f"  [pairs] {len(pairs)} PDF+QA")
    return pairs[:max_pdfs] if max_pdfs else pairs


# ===========================================================================
# 3. 코퍼스 구성 — base(hybrid)/EU + ablation 3종
# ===========================================================================

_converter = None


def get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        # 라이브러리의 make_converter()와 동일하게 명시 고정 — Docling 기본값에
        # 기대면 버전이 바뀔 때 표 구조 추출이 조용히 꺼질 수 있다.
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        _converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
        )
    return _converter


def chunk_page(c):
    try:
        for di in c.meta.doc_items:
            if di.prov:
                return di.prov[0].page_no
    except Exception:
        pass
    return None


def chunk_pages_all(c):
    """청크가 걸친 모든 페이지 집합 (PageHit@k 판정용)."""
    pages = set()
    try:
        for di in c.meta.doc_items:
            for p in (di.prov or []):
                pages.add(p.page_no)
    except Exception:
        pass
    return pages


def make_ablation_units(eu_list, drop_caption: bool, drop_context: bool):
    out = []
    for eu in eu_list:
        eu2 = copy.deepcopy(eu)
        if drop_caption:
            eu2.caption_text = None
        if drop_context:
            eu2.context_before = []
            eu2.context_after = []
        out.append(eu2)
    return out


def build_all_corpora(pdf_path: Path, doc_id: str):
    """PDF 1회 파싱으로 baseline + EU + ablation arm 전부 생성."""
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

    # base 팔의 표 청크에도 table_index를 매겨 TableHit@k를 양쪽 동일 기준으로
    # 판정한다(페이지 일치만으론 같은 페이지의 다른 표와 구분 불가). doc.tables[i]가
    # parsed.tables[i]와 동일 순서/객체라 identity로 매칭.
    base_table_index = [None] * len(all_chunks)
    for ci, c in enumerate(all_chunks):
        if not is_table_chunk(c):
            continue
        for ti, t in enumerate(doc.tables):
            if any(di is t for di in c.meta.doc_items):
                base_table_index[ci] = ti
                break

    non_table = [c for c in all_chunks if not is_table_chunk(c)]
    non_table_kept = filter_consumed_paragraphs(non_table, eu_list)

    def _pack(eu_variant):
        # TextChunk는 page_no 1개만 저장해 base(chunk_pages_all=전체 페이지)와
        # PageHit 기준이 비대칭해진다 — page_span을 명시로 채워 맞춘다.
        packed = []
        for i, c in enumerate(non_table_kept):
            tc = TextChunk(c, doc_id, i)
            tc.metadata["page_span"] = chunk_pages_all(c)
            packed.append(tc)
        return eu_variant + packed

    arms = {
        "base": {"chunks": all_chunks, "kind": "hybrid", "table_idx": base_table_index},
        "eu": {"chunks": _pack(eu_list), "kind": "eu"},
        "eu_no_caption": {"chunks": _pack(make_ablation_units(eu_list, True, False)), "kind": "eu"},
        "eu_no_context": {"chunks": _pack(make_ablation_units(eu_list, False, True)), "kind": "eu"},
        "eu_no_both": {"chunks": _pack(make_ablation_units(eu_list, True, True)), "kind": "eu"},
    }
    stats = {
        "n_tables": len(parsed.tables), "n_eu": len(eu_list),
        "n_split": sum(1 for eu in eu_list if eu.is_split),
        "n_baseline_chunks": len(all_chunks),
    }
    del doc
    gc.collect()
    return arms, stats


# ===========================================================================
# 4. 랭킹 및 PageHit@k / TableHit@k — 모든 arm에 동일 기준 적용
# ===========================================================================

def rank_indices(scores, k):
    return np.argsort(-np.asarray(scores), kind="stable")[:k].tolist()


class _MPDoc:
    """dedupe_by_chunk_id()용 최소 어댑터(.metadata만 필요)."""
    def __init__(self, idx, chunk_id):
        self.idx = idx
        self.metadata = {"chunk_id": chunk_id}


def top_k_with_maxpool(scores, chunk_ids, k, is_eu_kind):
    """EU 계열이면 max-pool dedup 적용, baseline이면 그냥 top-k."""
    wide = rank_indices(scores, max(k * 4, 20))
    if not is_eu_kind:
        return wide[:k]
    from evidence_chunker.export.langchain import dedupe_by_chunk_id
    candidates = [(_MPDoc(i, chunk_ids[i]), float(scores[i])) for i in wide]
    selected = dedupe_by_chunk_id(candidates, k=k)
    return [d.idx for d, _ in selected]


def page_hit_at_k(order, page_sets, gold_page):
    return {k: any(gold_page in page_sets[i] for i in order[:k]) for k in K_LIST}


def table_hit_at_k(order, table_idx_list, gold_table_index):
    """정답 표(gold_table_index) 자체를 top-k에서 찾았는가. gold가 None(표 없는 질문)이면 호출하지 않는다."""
    return {k: any(table_idx_list[i] == gold_table_index for i in order[:k]) for k in K_LIST}


# ===========================================================================
# 5. 동일 토큰 예산 + LLM 실답변 평가
# ===========================================================================

_gemini_client = None
_working_model = {"name": None}


def get_gemini_client(api_key: str):
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key,
                                       http_options=genai.types.HttpOptions(timeout=30_000))
    return _gemini_client


def ask_llm(context: str, question: str, api_key: Optional[str]) -> str:
    if not api_key:
        return ""
    client = get_gemini_client(api_key)
    prompt = ("Answer the question using only the context below. "
              "If the context does not contain the answer, say so explicitly.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}")
    cached = _working_model["name"]
    candidates = ([cached] if cached else []) + [m for m in MODEL_CANDIDATES if m != cached]
    RETRYABLE = ("503", "504", "429")
    attempts = []
    for model_name in candidates[:2]:
        for backoff in (0, 2, 4):
            if backoff:
                time.sleep(backoff)
            try:
                resp = client.models.generate_content(model=model_name, contents=prompt)
                _working_model["name"] = model_name
                return resp.text.strip()
            except Exception as e:
                err = str(e)
                attempts.append((model_name, err.split(chr(10))[0][:100]))
                if not any(code in err for code in RETRYABLE):
                    break
    return f"[LLM 호출 실패] {attempts[-1] if attempts else 'unknown'}"


def build_budget_context(order, texts, budget_tokens: int, tokenizer=None) -> str:
    def _count(s):
        return len(tokenizer.encode(s)) if tokenizer else len(s.split())

    parts, used = [], 0
    for i in order:
        t = texts[i]
        n = _count(t)
        if used + n > budget_tokens and parts:
            break
        parts.append(t)
        used += n
    return "\n\n---\n\n".join(parts)


def stratified_sample(rows, n, seed=LLM_SAMPLE_SEED):
    rng = np.random.default_rng(seed)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    total = len(rows)
    picked = []
    for t, group in by_type.items():
        take = max(1, round(n * len(group) / total)) if total else 0
        idx = rng.choice(len(group), size=min(take, len(group)), replace=False)
        picked += [group[i] for i in idx]
    return picked[:n]


def run_llm_eval(rows, all_arm_data, api_key: Optional[str], llm_sample_n: int, tokenizer=None):
    if not api_key:
        print("[LLM 평가] GEMINI_API_KEY 없음 — 건너뜀")
        return []

    sample = stratified_sample(rows, llm_sample_n)
    print(f"[LLM 평가] {len(sample)}문항 샘플로 baseline/EU 각각 채점")

    llm_rows = []
    for j, r in enumerate(sample, 1):
        ad = all_arm_data.get(r["doc_id"])
        if not ad or "base" not in ad or "eu" not in ad:
            continue
        b_order = r["arms"]["base"]["order"]
        e_order = r["arms"]["eu"]["order"]
        b_ctx = build_budget_context(b_order, ad["base"]["texts"], CONTEXT_TOKEN_BUDGET, tokenizer)
        e_disp = ad.get("eu__disp", ad["eu"]["texts"])
        e_ctx = build_budget_context(e_order, e_disp, CONTEXT_TOKEN_BUDGET, tokenizer)

        b_answer = ask_llm(b_ctx, r["question"], api_key)
        time.sleep(2)  # 무료 티어 분당 제한 대비
        e_answer = ask_llm(e_ctx, r["question"], api_key)
        time.sleep(2)

        llm_rows.append({
            "doc_id": r["doc_id"], "qid": r["qid"], "type": r["type"],
            "baseline_llm_correct": evidence_hit(r["answer_spec"], b_answer),
            "eu_llm_correct": evidence_hit(r["answer_spec"], e_answer),
            "baseline_answer": b_answer, "eu_answer": e_answer,
        })
        if j % 25 == 0:
            print(f"  [{j}/{len(sample)}]")
    return llm_rows


# ===========================================================================
# 6. 문서 1개 평가
# ===========================================================================

def evaluate_document(pdf_path: Path, qa_path: Path, doc_id: str, model, tokenizer=None):
    print(f"\n{'-'*62}\n  {doc_id}")
    try:
        arms, stats = build_all_corpora(pdf_path, doc_id)
    except Exception as e:
        print(f"  [ERR] {type(e).__name__}: {e}")
        return []

    print(f"  표 {stats['n_tables']} -> EU {stats['n_eu']} (분할 {stats['n_split']})  "
          f"baseline 청크 {stats['n_baseline_chunks']}")

    arm_data = {}
    for name, a in arms.items():
        chunks = a["chunks"]
        if not chunks:
            continue
        if a["kind"] == "hybrid":
            texts = [c.text for c in chunks]
            pages = [chunk_pages_all(c) for c in chunks]
            chunk_ids = [f"{doc_id}-hybrid-{i}" for i in range(len(chunks))]
            table_idx = a.get("table_idx") or [None] * len(chunks)
        else:
            # EU 계열: retrieval_units(행 단위)로 색인, 표시는 EU 전체 text
            units, ids, pages_u, table_idx_u, disp = [], [], [], [], []
            for c in chunks:
                page = c.metadata.get("page_no")
                page_span = c.metadata.get("page_span") or ({page} if page is not None else set())
                c_table_idx = c.metadata.get("table_index")
                for u in c.retrieval_units:
                    units.append(u)
                    ids.append(c.chunk_id)
                    pages_u.append(set(page_span) if page_span else {page})
                    table_idx_u.append(c_table_idx)
                    disp.append(c.text)
            texts, pages, chunk_ids, table_idx = units, pages_u, ids, table_idx_u
            arm_data[name + "__disp"] = disp
        if not texts:
            continue
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=ENCODE_BATCH)
        arm_data[name] = {"texts": texts, "pages": pages, "chunk_ids": chunk_ids,
                           "table_idx": table_idx, "emb": emb, "kind": a["kind"]}

    # 표 0개 회귀 검사(코퍼스 레벨): eu_list가 비면 eu 팔 텍스트/페이지는 base와
    # 완전히 같아야 한다. 불일치는 리트리버 차이가 아니라 채점 입력 자체가
    # 어긋났다는 신호.
    if stats["n_tables"] == 0 and "base" in arm_data and "eu" in arm_data:
        b_texts, e_texts = arm_data["base"]["texts"], arm_data["eu"]["texts"]
        b_pages, e_pages = arm_data["base"]["pages"], arm_data["eu"]["pages"]
        if b_texts != e_texts:
            print(f"  [회귀경고] {doc_id}: 표 0개인데 base/eu 텍스트 리스트 자체가 다름 "
                  f"(base={len(b_texts)}개 eu={len(e_texts)}개)")
        elif b_pages != e_pages:
            mismatches = [i for i in range(len(b_pages)) if b_pages[i] != e_pages[i]]
            print(f"  [회귀경고] {doc_id}: 표 0개인데 페이지 집합이 {len(mismatches)}개 청크에서 불일치 "
                  f"(예: idx={mismatches[0]} base={b_pages[mismatches[0]]} eu={e_pages[mismatches[0]]})")

    qa_list = json.load(open(qa_path, encoding="utf-8"))
    usable = [q for q in qa_list if q.get("question_delabeled") is not None
              and q.get("subset", "main") in (None, "main")]
    if not usable or "base" not in arm_data or "eu" not in arm_data:
        print("  [warn] 사용 가능 문항 0 또는 코퍼스 비어있음")
        return []

    q_emb = model.encode([q["question_delabeled"] for q in usable],
                          normalize_embeddings=True, show_progress_bar=False, batch_size=ENCODE_BATCH)

    rows = []
    for i, qa in enumerate(usable):
        gold_page = qa.get("page")
        row = {"doc_id": doc_id, "qid": qa.get("qid", ""), "type": qa.get("type", "unknown"),
               "question": qa["question_delabeled"], "expected_page": gold_page,
               "answer_spec": qa.get("answer_spec"), "answer": qa.get("answer", ""), "arms": {},
               "meta": qa.get("meta") or {}}

        gold_table_index = row["meta"].get("table_index")
        row["gold_table_index"] = gold_table_index

        for name, ad in arm_data.items():
            if name.endswith("__disp"):
                continue
            scores = np.dot(q_emb[i], ad["emb"].T)
            order = top_k_with_maxpool(scores, ad["chunk_ids"], TOP_K_MAX, ad["kind"] == "eu")
            page_at = page_hit_at_k(order, ad["pages"], gold_page)
            table_at = table_hit_at_k(order, ad["table_idx"], gold_table_index) if gold_table_index is not None else None
            top1_idx = order[0] if order else None
            disp = arm_data.get(name + "__disp")
            top1_text = (disp[top1_idx] if disp else ad["texts"][top1_idx]) if top1_idx is not None else ""
            row["arms"][name] = {
                "page_hit_at": page_at,
                "table_hit_at": table_at,
                "evidence_hit_at_1": evidence_hit(qa.get("answer_spec"), top1_text),
                "order": order, "texts_ref": name,
            }

        if stats["n_tables"] == 0 and "base" in row["arms"] and "eu" in row["arms"]:
            b_arm, e_arm = row["arms"]["base"], row["arms"]["eu"]
            if b_arm["page_hit_at"] != e_arm["page_hit_at"] or b_arm["evidence_hit_at_1"] != e_arm["evidence_hit_at_1"]:
                print(f"  [회귀경고] {doc_id}/{qa.get('qid','?')}: 표 0개인데 base/eu 판정 불일치 "
                      f"(base.page_hit={b_arm['page_hit_at']} eu.page_hit={e_arm['page_hit_at']} "
                      f"base.ev={b_arm['evidence_hit_at_1']} eu.ev={e_arm['evidence_hit_at_1']})")

        rows.append(row)

    n = len(rows)
    b1 = sum(r["arms"]["base"]["page_hit_at"][1] for r in rows) / n
    e1 = sum(r["arms"]["eu"]["page_hit_at"][1] for r in rows) / n
    print(f"  [{n}문항]  PageHit@1  base={b1:.3f}  eu={e1:.3f}")

    gc.collect()
    return rows, arm_data


# ===========================================================================
# 7. 실행 / 집계
# ===========================================================================

def run(pdf_dir: Path, qa_dir: Path, out_dir: Path, dev_only: bool, max_pdfs,
        gemini_api_key: Optional[str], llm_sample_n: int) -> dict:
    """전체 벤치마크를 실행하고 summary dict를 반환(+ 결과 JSON 저장)."""
    import evidence_chunker
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "dev20" if dev_only else "full90"
    print(f"[setup] device={device}  scope={tag}  evidence_chunker={evidence_chunker.__version__}")

    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    tokenizer = None
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        print("[warn] tiktoken 없음 — 토큰 예산을 단어수로 근사")

    pairs = pdf_qa_pairs(pdf_dir, qa_dir, dev_only, max_pdfs)
    if not pairs:
        raise SystemExit("[ERR] PDF-QA 쌍 없음 — --pdf-dir/--qa-dir 확인")

    all_rows, all_arm_data = [], {}
    for i, (pdf, qa, doc_id) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}]", end="")
        result = evaluate_document(pdf, qa, doc_id, model, tokenizer)
        if not result:
            continue
        rows, arm_data = result
        all_rows += rows
        all_arm_data[doc_id] = arm_data

    print(f"\n{'='*100}\n  HEADLINE — {len(all_rows)}문항 ({tag})\n{'='*100}")

    ARMS = ["base", "eu", "eu_no_caption", "eu_no_context", "eu_no_both"]
    for k in K_LIST:
        line = "  ".join(
            f"{a}={sum(r['arms'][a]['page_hit_at'][k] for r in all_rows if a in r['arms']) / max(1, sum(1 for r in all_rows if a in r['arms'])):.3f}"
            for a in ARMS
        )
        print(f"  PageHit@{k}:  {line}")
    ev_line = "  ".join(
        f"{a}={sum(r['arms'][a]['evidence_hit_at_1'] for r in all_rows if a in r['arms']) / max(1, sum(1 for r in all_rows if a in r['arms'])):.3f}"
        for a in ARMS
    )
    print(f"  evidence_hit@1:  {ev_line}")

    table_rows = [r for r in all_rows if r.get("gold_table_index") is not None]
    if table_rows:
        for k in K_LIST:
            line = "  ".join(
                f"{a}={sum(r['arms'][a]['table_hit_at'][k] for r in table_rows if a in r['arms'] and r['arms'][a].get('table_hit_at')) / max(1, sum(1 for r in table_rows if a in r['arms'] and r['arms'][a].get('table_hit_at'))):.3f}"
                for a in ARMS
            )
            print(f"  TableHit@{k}(표 질문 {len(table_rows)}개만):  {line}")
    else:
        print("  TableHit@k: 표 질문(meta.table_index 존재) 없음 — 건너뜀")

    doc_ids = [r["doc_id"] for r in all_rows if "base" in r["arms"] and "eu" in r["arms"]]
    b_vec = [int(r["arms"]["base"]["page_hit_at"][1]) for r in all_rows if "base" in r["arms"] and "eu" in r["arms"]]
    e_vec = [int(r["arms"]["eu"]["page_hit_at"][1]) for r in all_rows if "base" in r["arms"] and "eu" in r["arms"]]
    stats_page1 = paired_statistics(doc_ids, b_vec, e_vec) if doc_ids else None

    ablation_stats = {}
    for arm in ("eu_no_caption", "eu_no_context", "eu_no_both"):
        dids = [r["doc_id"] for r in all_rows if arm in r["arms"] and "eu" in r["arms"]]
        full_vec = [int(r["arms"]["eu"]["evidence_hit_at_1"]) for r in all_rows if arm in r["arms"] and "eu" in r["arms"]]
        ablated_vec = [int(r["arms"][arm]["evidence_hit_at_1"]) for r in all_rows if arm in r["arms"] and "eu" in r["arms"]]
        if dids:
            ablation_stats[arm] = paired_statistics(dids, ablated_vec, full_vec)  # ablated -> full (얼마나 잃었나)

    llm_rows = run_llm_eval(all_rows, all_arm_data, gemini_api_key, llm_sample_n, tokenizer)
    llm_summary = None
    if llm_rows:
        dids = [r["doc_id"] for r in llm_rows]
        b_llm = [int(r["baseline_llm_correct"]) for r in llm_rows]
        e_llm = [int(r["eu_llm_correct"]) for r in llm_rows]
        llm_summary = paired_statistics(dids, b_llm, e_llm)
        print(f"\n[LLM 실답변 평가] n={len(llm_rows)}  "
              f"baseline={llm_summary['baseline']:.3f}  eu={llm_summary['treatment']:.3f}  "
              f"diff={llm_summary['difference_pp']:+.1f}pp")

    def _blk(rs: list) -> Optional[dict]:
        """부분집합 요약 — base/eu 둘 다 페이지 일치 기준(page_hit_at[1])으로 통일."""
        rs = [r for r in rs if "base" in r["arms"] and "eu" in r["arms"]]
        if not rs:
            return None
        n = len(rs)
        return {
            "n": n,
            "baseline_recall_page": round(sum(r["arms"]["base"]["page_hit_at"][1] for r in rs) / n, 4),
            "eu_recall_page": round(sum(r["arms"]["eu"]["page_hit_at"][1] for r in rs) / n, 4),
            "baseline_evidence_hit": round(sum(r["arms"]["base"]["evidence_hit_at_1"] for r in rs) / n, 4),
            "eu_evidence_hit": round(sum(r["arms"]["eu"]["evidence_hit_at_1"] for r in rs) / n, 4),
        }

    by_type = {t: _blk([r for r in all_rows if r["type"] == t])
               for t in ("cell_value", "table_about", "context_dependent")}

    ctx_rows = [r for r in all_rows if r["type"] == "context_dependent"]
    context_dependent_slices = {
        "in_window": _blk([r for r in ctx_rows if (r["meta"].get("dist_pt") or 0) <= BBOX_THRESHOLD]),
        "out_window": _blk([r for r in ctx_rows if (r["meta"].get("dist_pt") or 0) > BBOX_THRESHOLD]),
        "explicit_ref": _blk([r for r in ctx_rows if r["meta"].get("ctx_explicit_ref") is True]),
        "no_explicit_ref": _blk([r for r in ctx_rows if r["meta"].get("ctx_explicit_ref") is False]),
    } if ctx_rows else {}

    meta_slices = {
        "cross_table": _blk([r for r in all_rows if (r["meta"].get("n_tables_on_page") or 0) >= 2]),
        "single_table": _blk([r for r in all_rows if (r["meta"].get("n_tables_on_page") or 0) == 1]),
        "toc_zone": _blk([r for r in all_rows if (r["meta"].get("page_index") or 1) <= 0.1]),
        "multi_header": _blk([r for r in all_rows if (r["meta"].get("header_rows") or 1) > 1]),
    }

    summary = {
        "config": {"scope": tag, "bbox_threshold": BBOX_THRESHOLD, "sim_threshold": SIM_THRESHOLD,
                   "embed_model": EMBED_MODEL_NAME, "context_token_budget": CONTEXT_TOKEN_BUDGET,
                   "llm_sample_n": len(llm_rows), "evidence_chunker": evidence_chunker.__version__},
        "n_questions": len(all_rows),
        "page_hit_at_k": {k: {a: round(sum(r["arms"][a]["page_hit_at"][k] for r in all_rows if a in r["arms"]) /
                                        max(1, sum(1 for r in all_rows if a in r["arms"])), 4)
                               for a in ARMS} for k in K_LIST},
        "evidence_hit_at_1": {a: round(sum(r["arms"][a]["evidence_hit_at_1"] for r in all_rows if a in r["arms"]) /
                                        max(1, sum(1 for r in all_rows if a in r["arms"])), 4) for a in ARMS},
        "table_hit_at_k": {k: {a: round(sum(r["arms"][a]["table_hit_at"][k] for r in table_rows if a in r["arms"] and r["arms"][a].get("table_hit_at")) /
                                         max(1, sum(1 for r in table_rows if a in r["arms"] and r["arms"][a].get("table_hit_at"))), 4)
                               for a in ARMS} for k in K_LIST} if table_rows else None,
        "n_table_questions": len(table_rows),
        "by_type": by_type,
        "context_dependent_slices": context_dependent_slices,
        "meta_slices": meta_slices,
        "paired_statistics_page_hit_1_base_vs_eu": stats_page1,
        "ablation_paired_statistics": ablation_stats,
        "llm_answer_eval": {"summary": llm_summary, "rows": llm_rows},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"bench_v6_{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"bench_v6_{tag}_rows.json").write_text(
        json.dumps([{**r, "arms": {a: {k: v for k, v in d.items() if k != "order"} for a, d in r["arms"].items()}}
                    for r in all_rows], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out_dir}/bench_v6_{tag}.json")
    return summary


# ===========================================================================
# 8. CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--pdf-dir", type=Path, required=True, help="벤치마크 PDF 디렉토리")
    p.add_argument("--qa-dir", type=Path, required=True,
                   help="generate_qa_docling.py가 만든 {문서}_qa.json이 있는 디렉토리")
    p.add_argument("--out-dir", type=Path, required=True, help="결과 JSON을 저장할 디렉토리")
    p.add_argument("--dev-only", action="store_true", help="dev 서브셋(20개 문서)만 실행")
    p.add_argument("--max-pdfs", type=int, default=None, help="추가 상한 (디버깅용)")
    p.add_argument("--gemini-api-key", type=str, default=None,
                   help="LLM 실답변 평가용. 미지정 시 GEMINI_API_KEY 환경변수 사용, "
                        "둘 다 없으면 그 단계만 건너뜀")
    p.add_argument("--llm-sample-n", type=int, default=LLM_SAMPLE_N,
                   help="LLM 실답변 평가 샘플 문항 수(유형별 층화)")
    p.add_argument("--bbox-threshold", type=float, default=None, help="BBOX_THRESHOLD")
    p.add_argument("--sim-threshold", type=float, default=None, help="SIM_THRESHOLD")
    p.add_argument("--embed-model", type=str, default=None, help="EMBED_MODEL_NAME")
    p.add_argument("--encode-batch", type=int, default=None, help="ENCODE_BATCH")
    p.add_argument("--context-token-budget", type=int, default=None, help="CONTEXT_TOKEN_BUDGET")
    return p.parse_args()


def main() -> None:
    global BBOX_THRESHOLD, SIM_THRESHOLD, EMBED_MODEL_NAME, ENCODE_BATCH, CONTEXT_TOKEN_BUDGET

    args = parse_args()

    if args.bbox_threshold is not None:
        BBOX_THRESHOLD = args.bbox_threshold
    if args.sim_threshold is not None:
        SIM_THRESHOLD = args.sim_threshold
    if args.embed_model is not None:
        EMBED_MODEL_NAME = args.embed_model
    if args.encode_batch is not None:
        ENCODE_BATCH = args.encode_batch
    if args.context_token_budget is not None:
        CONTEXT_TOKEN_BUDGET = args.context_token_budget

    gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")

    run(args.pdf_dir, args.qa_dir, args.out_dir, args.dev_only, args.max_pdfs,
        gemini_api_key, args.llm_sample_n)


if __name__ == "__main__":
    main()
