# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
generate_qa_docling.py — 벤치마크용 Q&A 자동 생성기 (자기참조 0, 결정적)

설계 배경 전체는 wiki의 QA-Generation 참고.

사용법:
    python generate_qa_docling.py --pdf-dir ./data/pdfs --out-dir ./auto_qa
    python generate_qa_docling.py --pdf-dir ./data/pdfs --out-dir ./auto_qa --limit 3   # 소량 검증

주의: 평가 대상 패키지(evidence_chunker 등)를 sys.path에 넣지 말 것 (격리 위반).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

GENERATOR_VERSION = "3.4.0-docling-shared"
DEFAULT_ARTIFACTS_PATH: str | None = None  # None이면 HuggingFace Hub에서 자동 다운로드

MAX_CELL_QA_PER_TABLE = 2
MAX_CTX_QA_PER_TABLE = 2
MAX_LABEL_LEN = 120
MIN_TOKEN_LEN = 3
N_CONTEXT_KEYS = 2
MIN_CTX_PARA_CHARS = 80
MAX_QUESTION_LEN = 200
MIN_LINK_TOKENS = 2  # 표-문단 공통 숫자 토큰 최소 개수 (우연 일치 방지)

_TABLE_REF_RE = re.compile(r"(table|tab\.|표)\s*\.?\s*[A-Z]?\.?\s*\d+", re.IGNORECASE)
CONTEXT_LABELS = {"text", "paragraph", "list_item"}

# 이 프로젝트의 평가 대상 모듈만 금지한다 (Docling은 두 arm 공통 프론트엔드라 허용).
FORBIDDEN_MODULES = {
    "smart_chunker", "interfaces", "context_attacher", "table_splitter",
    "token_counter", "bbox_utils", "langchain_wrapper",
    "_table_utils", "_caption_mapper", "evidence_chunker",
}


# ===========================================================================
# 1. 정규화 / 채점
# ===========================================================================
# normalize_for_em / has_token / em_hit은 실제 채점 스크립트(benchmarks/measure_recall_*.py)
# 와 동일한 규칙을 유지해야 한다 — 어긋나면 answer_spec이 의미를 잃는다.

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
    """값 + 모든 context_keys가 같은 청크 안에 있으면 정답 (집합 포함 검사)."""
    c = normalize_for_em(chunk_text)
    if not has_token(spec["value"], c):
        return False
    return all(has_token(k, c) for k in spec["context_keys"])


_ALNUM_RE = re.compile(r"[0-9a-zÀ-ɏ가-힣]")


def tokens_of(s: str) -> list[str]:
    return normalize_for_em(s).split()


def useful_tokens(s: str) -> list[str]:
    out = []
    for t in tokens_of(s):
        t = t.rstrip(".")
        if len(t) >= MIN_TOKEN_LEN and _ALNUM_RE.search(t):
            out.append(t)
    return out


_LEADING_JUNK = "(),;:|/"


def is_sane_value(v: str) -> bool:
    """잘린 셀 조각("(Dec", "et al.,")을 정답 value로 쓰지 않기 위한 위생 검사."""
    n = (v or "").strip()
    if not n or not _ALNUM_RE.search(normalize_for_em(n)):
        return False
    if n[0] in _LEADING_JUNK:
        return False
    for open_ch, close_ch in (("(", ")"), ("[", "]"), ("{", "}")):
        if n.count(open_ch) != n.count(close_ch):
            return False  # 괄호 불균형 = 셀이 잘렸다는 신호
    return True


def clean(v) -> str:
    return " ".join(str(v or "").split())


# 질문 라벨용 위생 처리. answer_spec.value 자체는 건드리지 않는다
# (표 markdown 원문과 일치해야 CI 불변식이 성립).
_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")          # [9] [1, 2]
_CITEKEY_RE = re.compile(r"\[[A-Za-z][A-Za-z+.\s]{0,14}\d{2}[a-z]?\]")  # [Ant24] [HBK+21b]
_MARKER_RE = re.compile(r"[†‡§※#]")


def clean_label(s: str) -> str:
    if not s:
        return ""
    s = _CITATION_RE.sub("", s)
    s = _CITEKEY_RE.sub("", s)
    s = _MARKER_RE.sub("", s)
    s = re.sub(r"\s+([,.;:)\]])", r"\1", s)
    return " ".join(s.split())


_NUM_GAP_RE = re.compile(r"(\d)\s*\.\s*(\d)")


def display_answer(v: str) -> str:
    """표시·검수용 answer. PDF 렌더링에 가깝게 정리(채점에는 미사용)."""
    a = clean_label(v)
    for _ in range(3):
        a = _NUM_GAP_RE.sub(r"\1.\2", a)
    return a


# 순수 숫자 헤더 = 데이터 행이 헤더로 잘못 승격된 것 (연도는 예외로 보존).
_PURE_NUM_HEADER_RE = re.compile(r"^[\d.,%\s]+$")
_YEAR_HEADER_RE = re.compile(r"^(19|20)\d{2}$")


def value_counts_in_table(grid: list[list[str]]) -> Counter:
    """표 데이터 셀 값 빈도. 정답이 유일하지 않으면 문항 후보에서 제외하는 데 씀."""
    c: Counter = Counter()
    if len(grid) < 2:
        return c
    ncol = len(grid[0])
    for r in range(1, len(grid)):
        for cc in range(1, min(ncol, len(grid[r]))):
            v = normalize_for_em(grid[r][cc])
            if v:
                c[v] += 1
    return c


def is_numeric_header(h: str) -> bool:
    h = (h or "").strip()
    if _YEAR_HEADER_RE.match(h):
        return False
    return bool(_PURE_NUM_HEADER_RE.match(h))


# ===========================================================================
# 2. Docling 접근 — raw 필드만. 우리 유틸을 재사용하지 않는다.
# ===========================================================================

def make_converter(artifacts_path: str | None):
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    if artifacts_path:
        opts.artifacts_path = artifacts_path
    return DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opts)})


def build_grid(cells: list[dict], num_rows: int, num_cols: int) -> tuple[list[list[str]], int]:
    """table_cells → (그리드, 헤더행수). grid[0]은 조립된 헤더 행, grid[1:]는 데이터.

    span 전파 → 헤더 행수 판정 → 캡션 행 제외 → 세로 조인 순으로 처리한다.
    evidence_chunker.flatten.build_col_header_map()과는 별개 구현(값은 유사)이다.
    """
    if num_rows <= 0 or num_cols <= 0:
        return [], 0

    raw = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in cells:
        r = cell.get("start_row_offset_idx", 0)
        c = cell.get("start_col_offset_idx", 0)
        r_end = cell.get("end_row_offset_idx", r + 1)
        c_end = cell.get("end_col_offset_idx", c + 1)
        txt = clean(cell.get("text"))
        if not txt:
            continue
        for rr in range(max(0, r), min(r_end, num_rows)):
            for cc in range(max(0, c), min(c_end, num_cols)):
                if not raw[rr][cc]:
                    raw[rr][cc] = txt

    header_rows = 0
    for cell in cells:
        if cell.get("column_header"):
            header_rows = max(
                header_rows,
                cell.get("end_row_offset_idx", cell.get("start_row_offset_idx", 0) + 1),
            )
    if header_rows <= 0:
        header_rows = 1  # column_header 플래그 없음 → 첫 행을 헤더로 폴백
    header_rows = min(header_rows, max(1, num_rows - 1))

    # 전 열이 동일한 텍스트인 행 = 캡션이 표 첫 행으로 통째로 span된 경우.
    # 캡션을 열 헤더로 조립하면 캡션 매핑(평가 대상 모듈)이 뒷문으로 질문에 샌다.
    caption_rows: set[int] = set()
    if num_cols >= 2:
        for rr in range(header_rows):
            vals = [raw[rr][cc] for cc in range(num_cols)]
            if vals[0] and all(v == vals[0] for v in vals):
                caption_rows.add(rr)
    if caption_rows and len(caption_rows) == header_rows and header_rows < num_rows - 1:
        header_rows += 1  # 헤더 행이 전부 캡션이면 다음 행을 실제 헤더로 승격

    header: list[str] = []
    for cc in range(num_cols):
        parts: list[str] = []
        for rr in range(header_rows):
            if rr in caption_rows:
                continue
            t = raw[rr][cc]
            if t and t not in parts:
                parts.append(t)
        header.append(" ".join(parts))

    return [header] + raw[header_rows:], header_rows


def table_markdown(doc, table) -> str:
    """HybridChunker가 청크에 담는 바로 그 텍스트 (CI 불변식의 기준)."""
    for call in (lambda: table.export_to_markdown(doc), lambda: table.export_to_markdown()):
        try:
            md = call()
            if md:
                return md
        except Exception:
            continue
    return ""


def page_paragraphs(doc) -> dict[int, list[dict]]:
    """페이지별 본문 문단 + bbox. bbox는 게이트가 아니라 거리 측정(meta.dist_pt)용."""
    out: dict[int, list[dict]] = defaultdict(list)
    for item in getattr(doc, "texts", []) or []:
        try:
            d = item.model_dump()
        except Exception:
            continue
        if d.get("label") not in CONTEXT_LABELS:
            continue
        prov = d.get("prov") or []
        if not prov:
            continue
        pg = prov[0].get("page_no", -1)
        txt = clean(d.get("text"))
        if pg != -1 and len(txt) >= MIN_CTX_PARA_CHARS:
            out[pg].append({"text": txt, "bbox": prov[0].get("bbox") or {}})
    return out


def vertical_gap_pt(table_bbox: dict, para_bbox: dict) -> float | None:
    """표-문단 수직 간격(PDF 포인트). Docling bbox는 BOTTOMLEFT라 t > b."""
    if not table_bbox or not para_bbox:
        return None
    try:
        t_top, t_bot = float(table_bbox["t"]), float(table_bbox["b"])
        p_cy = (float(para_bbox["t"]) + float(para_bbox["b"])) / 2.0
    except (KeyError, TypeError, ValueError):
        return None
    if p_cy > t_top:
        return round(p_cy - t_top, 1)
    if p_cy < t_bot:
        return round(t_bot - p_cy, 1)
    return 0.0


# ===========================================================================
# 3. context_keys 선정 — 문서 내 희소 토큰 우선 (결정적, RNG 없음)
# ===========================================================================

def pick_context_keys(sources: list[str], doc_counts: Counter,
                       exclude: set[str], n: int = N_CONTEXT_KEYS) -> list[str]:
    cands: list[str] = []
    seen: set[str] = set()
    for s in sources:
        for t in useful_tokens(s):
            if t in exclude or t in seen:
                continue
            seen.add(t)
            cands.append(t)
    if not cands:
        return []
    cands.sort(key=lambda t: (doc_counts.get(t, 1), -len(t), t))
    return cands[:n]


# ===========================================================================
# 4. 문항 생성
# ===========================================================================

def _qa(qid, qtype, question, answer, value, context_keys, table_ref, tbl, delabeled=None):
    return {
        "doc_id": tbl["doc_id"],
        "qid": f'{tbl["doc_id"]}::{qid}',  # 문서 식별자 접두사 없으면 여러 문서에서 qid 충돌
        "type": qtype,
        "question": question,
        "question_delabeled": delabeled if delabeled is not None else question,
        "answer": answer,  # 표시용, 채점에 쓰지 않음
        "answer_spec": {"value": value, "context_keys": list(context_keys)},
        "table_ref": table_ref,
        "page": tbl["page"],
        "table_type": "auto_generated",
        "subset": "main",
        "meta": {
            "table_index": tbl["table_index"],
            "n_tables_on_page": tbl["n_tables_on_page"],
            "page_index": tbl["page_index"],
            "n_rows": tbl["num_rows"],
            "n_cols": tbl["num_cols"],
            "header_rows": tbl["header_rows"],
        },
    }


def gen_cell_value(tbl, doc_counts, table_ref, qid_prefix, skips) -> list[dict]:
    grid = tbl["grid"]
    headers = grid[0]
    out, seen = [], set()
    vcounts = value_counts_in_table(grid)

    cands = []
    for r in range(1, len(grid)):
        row_label = grid[r][0]
        if not row_label:
            skips["cell:no_row_label"] += 1
            continue
        if len(row_label) > MAX_LABEL_LEN:
            skips["cell:row_label_too_long"] += 1
            continue
        for c in range(1, len(headers)):
            value, col_header = grid[r][c], headers[c]
            if not value or not is_sane_value(value):
                skips["cell:unusable_value"] += 1
                continue
            if not col_header:
                skips["cell:no_col_header"] += 1
                continue
            if len(col_header) > MAX_LABEL_LEN:
                skips["cell:col_header_too_long"] += 1
                continue
            if is_numeric_header(col_header):
                skips["cell:numeric_col_header"] += 1
                continue
            if vcounts[normalize_for_em(value)] > 1:
                skips["cell:value_not_unique_in_table"] += 1
                continue
            rarity = min([doc_counts.get(t, 1)
                          for t in useful_tokens(row_label) + useful_tokens(col_header)] or [10 ** 6])
            cands.append((rarity, row_label, col_header, value))

    cands.sort(key=lambda x: (x[0], x[1], x[2], x[3]))  # RNG 없이 완전 결정적

    for _, row_label, col_header, value in cands:
        if len(out) >= MAX_CELL_QA_PER_TABLE:
            break
        keys = pick_context_keys([row_label, col_header], doc_counts,
                                  exclude=set(useful_tokens(value)))
        if len(keys) < N_CONTEXT_KEYS:
            skips["cell:not_enough_keys"] += 1
            continue
        sig = (normalize_for_em(value), tuple(sorted(keys)))
        if sig in seen:
            skips["cell:duplicate_spec"] += 1
            continue
        seen.add(sig)
        body = f"what is the {clean_label(col_header)} for {clean_label(row_label)}?"
        if len(body) > MAX_QUESTION_LEN:
            skips["cell:question_too_long"] += 1
            continue
        if has_token(value, normalize_for_em(body)):
            skips["cell:answer_leak_in_question"] += 1  # 라벨 자체에 정답값 포함
            continue
        out.append(_qa(f"{qid_prefix}_c{len(out)}", "cell_value",
                        f"In {table_ref}, {body}", display_answer(value), value,
                        keys, table_ref, tbl,
                        delabeled=body[0].upper() + body[1:]))
    return out


def gen_table_about(tbl, doc_counts, table_ref, qid_prefix, skips) -> list[dict]:
    # 같은 페이지에 표가 2개 이상이면 "what columns does it have?"가 어느 표인지
    # table_ref만으로 특정 불가능 → 생성하지 않음.
    if tbl.get("n_tables_on_page", 1) >= 2:
        skips["about:ambiguous_page"] += 1
        return []
    headers = [h for h in tbl["grid"][0] if h and len(h) <= MAX_LABEL_LEN]
    cols = []
    for h in headers:
        if not cols or cols[-1] != h:  # 병합 헤더의 인접 중복만 제거
            cols.append(h)
    if len(cols) < 3:
        skips["about:too_few_columns"] += 1
        return []

    ranked = sorted(cols, key=lambda h: (min([doc_counts.get(t, 1)
                                               for t in useful_tokens(h)] or [10 ** 6]), h))
    value_col = ranked[0]
    keys = pick_context_keys(ranked[1:], doc_counts, exclude=set(useful_tokens(value_col)))
    if len(keys) < N_CONTEXT_KEYS:
        skips["about:not_enough_keys"] += 1
        return []

    # keys는 채점(answer_spec.context_keys)에만 쓰이고 질문 문장에는 노출되지 않는다.
    body = "what columns does it have?"
    return [_qa(f"{qid_prefix}_cols", "table_about", f"In {table_ref}, {body}",
                ", ".join(clean_label(c) for c in cols), value_col, keys, table_ref, tbl,
                delabeled=body[0].upper() + body[1:])]


def gen_context_dependent(tbl, paragraphs, doc_counts, table_ref, qid_prefix, skips) -> list[dict]:
    """
    v1 = 표 ∩ 문단 (연결 증거, 문항엔 안 씀)  v2 = 표 - 문단 (정답)  k = 문단 - 표 (context_keys)
    표만 있으면 k 없어 실패, 문단만 있으면 v2 없어 실패, 표+문단(EU)만 둘 다 있어 통과.
    """
    grid = tbl["grid"]
    headers = grid[0]
    vcounts = value_counts_in_table(grid)
    table_tokens: set[str] = set()
    for row in grid:
        table_tokens.update(tokens_of(" ".join(row)))

    out = []
    used_values: set[str] = set()  # 같은 셀 재사용 시 문항 간 상관 발생 방지
    # 거리 게이트 없음 — 탐색 범위는 페이지 전체(모듈의 300pt를 진부분집합으로 포함).
    # 거리는 meta.dist_pt로 측정만 한다.
    for para_item in paragraphs:
        if len(out) >= MAX_CTX_QA_PER_TABLE:
            break
        para = para_item["text"]
        para_norm = normalize_for_em(para)
        para_tokens = set(tokens_of(para))

        link = [t for t in (table_tokens & para_tokens) if any(ch.isdigit() for ch in t)]
        explicit_ref = bool(_TABLE_REF_RE.search(para))
        if not link:
            skips["ctx:no_link_value"] += 1
            continue
        if len(link) < MIN_LINK_TOKENS and not explicit_ref:
            skips["ctx:weak_link_signal"] += 1
            continue

        pool = [t for t in useful_tokens(para) if t not in table_tokens]
        keys = pick_context_keys([" ".join(dict.fromkeys(pool))], doc_counts, exclude=set())
        if len(keys) < N_CONTEXT_KEYS:
            skips["ctx:not_enough_para_keys"] += 1
            continue

        v2 = v2_row = v2_col = ""
        for r in range(1, len(grid)):
            row_label = grid[r][0]
            if not row_label or len(row_label) > MAX_LABEL_LEN:
                continue
            for c in range(1, len(headers)):
                value, col_header = grid[r][c], headers[c]
                if not value or not col_header or len(col_header) > MAX_LABEL_LEN:
                    continue
                if not is_sane_value(value):
                    continue
                if has_token(value, para_norm):
                    continue  # 문단에도 있으면 v2 자격 없음
                if normalize_for_em(value) in used_values:
                    continue
                if vcounts[normalize_for_em(value)] > 1:
                    continue
                v2, v2_row, v2_col = value, row_label, col_header
                break
            if v2:
                break
        if not v2:
            skips["ctx:no_table_only_value"] += 1
            continue
        used_values.add(normalize_for_em(v2))

        body = (f"In the context of {keys[0]} and {keys[1]}, "
                f"what is the {clean_label(v2_col)} for {clean_label(v2_row)}?")
        if len(body) > MAX_QUESTION_LEN:
            skips["ctx:question_too_long"] += 1
            continue
        qa = _qa(f"{qid_prefix}_ctx{len(out)}", "context_dependent", body,
                  display_answer(v2), v2, keys, table_ref, tbl)
        qa["meta"]["ctx_link_tokens"] = sorted(link)[:5]
        qa["meta"]["ctx_explicit_ref"] = explicit_ref
        qa["meta"]["ctx_para_excerpt"] = para[:200]
        qa["meta"]["dist_pt"] = vertical_gap_pt(tbl.get("bbox") or {}, para_item.get("bbox") or {})
        qa["_para"] = para  # 검증용, 출력 직전 제거
        out.append(qa)
    return out


# ===========================================================================
# 5. CI 불변식 — 내보내기 전 자체 검증
# ===========================================================================
#   유형                       표 markdown 단독   문단 단독   표+문단
#   cell_value / table_about   PASS               -           PASS
#   context_dependent          FAIL               FAIL        PASS

def validate(qa: dict, table_md: str) -> tuple[bool, str]:
    spec = qa["answer_spec"]
    if qa["type"] in ("cell_value", "table_about"):
        return (True, "ok") if em_hit(spec, table_md) else (False, "fail_on_table_markdown")

    para = qa.get("_para", "")
    if em_hit(spec, table_md):
        return False, "ctx_passes_on_table_alone"
    if em_hit(spec, para):
        return False, "ctx_passes_on_para_alone"
    if not em_hit(spec, table_md + "\n" + para):
        return False, "ctx_fails_on_combined"
    return True, "ok"


# ===========================================================================
# 6. PDF 1개 처리
# ===========================================================================

def generate_for_doc(doc, doc_id: str) -> tuple[list[dict], Counter, Counter]:
    skips: Counter = Counter()
    rejects: Counter = Counter()

    # 문서 전체 토큰 빈도를 생성 전에 완성 (표별로 갱신하면 앞쪽 표일수록 희소도 왜곡).
    doc_counts: Counter = Counter()
    for item in getattr(doc, "texts", []) or []:
        try:
            doc_counts.update(tokens_of(item.model_dump().get("text") or ""))
        except Exception:
            pass
    for table in getattr(doc, "tables", []) or []:
        try:
            d = table.model_dump()
            data = d.get("data") or {}
            grid, _ = build_grid(data.get("table_cells") or [],
                                  data.get("num_rows", 0), data.get("num_cols", 0))
            for row in grid:
                doc_counts.update(tokens_of(" ".join(row)))
        except Exception:
            pass

    paras_by_page = page_paragraphs(doc)

    tables_meta = []
    n_pages = max(1, len(getattr(doc, "pages", {}) or {1: None}))
    page_tbl_count: Counter = Counter()
    for table in getattr(doc, "tables", []) or []:
        try:
            d = table.model_dump()
        except Exception:
            continue
        prov = d.get("prov") or []
        pg = prov[0].get("page_no", -1) if prov else -1
        bbox = prov[0].get("bbox") or {} if prov else {}
        if pg != -1:
            page_tbl_count[pg] += 1
        tables_meta.append((table, d, pg, bbox))

    all_qa: list[dict] = []
    per_page_seq: Counter = Counter()
    seen_doc_specs: set = set()   # Docling이 같은 표를 여러 TableItem으로 인식하는 경우 대비
    seen_questions: set = set()   # 문서 내 동일 질문(같은 top-1) 억제

    for table_index, (table, d, pg, tbbox) in enumerate(tables_meta):
        if pg == -1:
            skips["table:no_prov"] += 1
            continue
        data = d.get("data") or {}
        cells = data.get("table_cells") or []
        num_rows, num_cols = data.get("num_rows", 0), data.get("num_cols", 0)
        grid, header_rows = build_grid(cells, num_rows, num_cols)
        if len(grid) < 2 or len(grid[0]) < 2:
            skips["table:grid_too_small"] += 1
            continue

        idx = per_page_seq[pg]
        per_page_seq[pg] += 1
        tbl = {
            "grid": grid, "page": pg, "table_index": table_index,
            "bbox": tbbox, "doc_id": doc_id, "header_rows": header_rows,
            "num_rows": num_rows, "num_cols": num_cols,
            "n_tables_on_page": page_tbl_count[pg],
            "page_index": round(pg / n_pages, 3),
        }
        qid_prefix = f"auto_p{pg}_t{idx}"
        table_ref = f"the table on page {pg}"  # 캡션은 읽지 않음 — 캡션 매핑이 평가 대상 모듈

        md = table_markdown(doc, table)
        if not md:
            skips["table:no_markdown"] += 1
            continue

        batch = []
        batch += gen_cell_value(tbl, doc_counts, table_ref, qid_prefix, skips)
        batch += gen_table_about(tbl, doc_counts, table_ref, qid_prefix, skips)
        batch += gen_context_dependent(tbl, paras_by_page.get(pg, []), doc_counts,
                                        table_ref, qid_prefix, skips)

        for qa in batch:
            ok, reason = validate(qa, md)
            qa.pop("_para", None)
            if not ok:
                rejects[f"{qa['type']}:{reason}"] += 1
                continue
            spec = qa["answer_spec"]
            sig = (qa["type"], normalize_for_em(spec["value"]),
                   tuple(sorted(spec["context_keys"])))
            if sig in seen_doc_specs:
                rejects[f"{qa['type']}:duplicate_in_document"] += 1
                continue
            if qa["question"] in seen_questions:
                rejects[f"{qa['type']}:duplicate_question"] += 1
                continue
            seen_doc_specs.add(sig)
            seen_questions.add(qa["question"])
            all_qa.append(qa)

    return all_qa, skips, rejects


# ===========================================================================
# 7. 격리 검증
# ===========================================================================

def assert_isolation() -> None:
    leaked = {m for m in sys.modules if m.split(".")[0] in FORBIDDEN_MODULES}
    if leaked:
        raise RuntimeError(
            f"격리 위반: 평가 대상 모듈이 로드됨 -> {sorted(leaked)}\n"
            "이 스크립트는 평가 대상 라이브러리를 import하면 안 됩니다 (docling은 허용)."
        )
    print("[isolation] OK — 평가 대상 모듈이 로드되지 않았습니다.")


# ===========================================================================
# 8. main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--pdf-dir", type=Path, required=True, help="QA를 생성할 PDF들이 있는 디렉토리")
    p.add_argument("--out-dir", type=Path, required=True, help="생성된 QA JSON을 저장할 디렉토리")
    p.add_argument("--limit", type=int, default=None, help="먼저 소량(예: 3)으로 검증, 생략 시 전체 실행")
    p.add_argument("--artifacts-path", type=str, default=DEFAULT_ARTIFACTS_PATH,
                    help="Docling 로컬 모델 아티팩트 경로 (없으면 HuggingFace Hub에서 다운로드)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert_isolation()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))
    if args.limit:
        pdf_paths = pdf_paths[:args.limit]
    print(f"[setup] {len(pdf_paths)} PDF(s) queued (limit={args.limit})")

    converter = make_converter(args.artifacts_path)
    merged: list[dict] = []
    per_file: dict[str, int] = {}
    skip_report: dict[str, dict] = {}
    reject_report: dict[str, dict] = {}
    type_counter: Counter = Counter()
    total_rejects: Counter = Counter()

    for i, pdf_path in enumerate(pdf_paths, 1):
        try:
            doc = converter.convert(str(pdf_path)).document
            qa, skips, rejects = generate_for_doc(doc, pdf_path.stem)
        except Exception as e:
            print(f"  [{i}/{len(pdf_paths)}] {pdf_path.name}: ERROR {type(e).__name__}: {e}")
            skip_report[pdf_path.stem] = {"__error__": f"{type(e).__name__}: {e}"}
            per_file[pdf_path.stem] = 0
            continue

        (args.out_dir / f"{pdf_path.stem}_qa.json").write_text(
            json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

        merged += qa
        per_file[pdf_path.stem] = len(qa)
        skip_report[pdf_path.stem] = dict(skips)
        reject_report[pdf_path.stem] = dict(rejects)
        total_rejects.update(rejects)
        type_counter.update(q["type"] for q in qa)

        tc = Counter(q["type"] for q in qa)
        print(f"  [{i}/{len(pdf_paths)}] {pdf_path.stem}: {len(qa)}문항 "
              f"(cell={tc['cell_value']} about={tc['table_about']} ctx={tc['context_dependent']})"
              f"{'  <-- 0건' if not qa else ''}")

    if not merged:
        raise RuntimeError("생성된 문항이 0개입니다. --pdf-dir / Docling 설치를 확인하세요.")

    (args.out_dir / "_all_auto_qa.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    (args.out_dir / "_generation_meta.json").write_text(json.dumps({
        "generator": "generate_qa_docling.py",
        "generator_version": GENERATOR_VERSION,
        "parser": "docling (두 arm 공통 프론트엔드)",
        "isolation": "평가 대상 라이브러리 import 없음 (assert_isolation 통과)",
        "ci_invariant": {
            "cell_value/table_about": "표 markdown에서도 em_hit PASS",
            "context_dependent": "표 단독 FAIL / 문단 단독 FAIL / 합계 PASS",
        },
        "config": {
            "MAX_CELL_QA_PER_TABLE": MAX_CELL_QA_PER_TABLE,
            "MAX_CTX_QA_PER_TABLE": MAX_CTX_QA_PER_TABLE,
            "MAX_LABEL_LEN": MAX_LABEL_LEN,
            "N_CONTEXT_KEYS": N_CONTEXT_KEYS,
            "MIN_CTX_PARA_CHARS": MIN_CTX_PARA_CHARS,
        },
        "totals": {"pdfs": len(pdf_paths), "questions": len(merged), **type_counter},
        "validation_rejects": dict(total_rejects),
        "per_file": per_file,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (args.out_dir / "_skip_reasons.json").write_text(
        json.dumps(skip_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "_validation.json").write_text(
        json.dumps(reject_report, ensure_ascii=False, indent=2), encoding="utf-8")

    zero = [k for k, v in per_file.items() if v == 0]
    print("\n" + "=" * 64)
    print(f"  총 {len(merged)}문항 / {len(pdf_paths)}개 PDF")
    print(f"  cell_value={type_counter['cell_value']}  "
          f"table_about={type_counter['table_about']}  "
          f"context_dependent={type_counter['context_dependent']}")
    print(f"  검증 폐기: {dict(total_rejects) or '없음'}")
    if zero:
        print(f"  0건 PDF {len(zero)}개 -> _skip_reasons.json 확인: {zero[:10]}")
    print(f"  출력: {args.out_dir}")
    print("=" * 64)
    assert_isolation()


if __name__ == "__main__":
    main()
