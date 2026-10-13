# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
DoclingParser의 BOTTOMLEFT -> TOPLEFT bbox 변환이 맞는지 잠그는 테스트.

이 변환이 이번 파서 추상화에서 가장 위험한 부분이다 — 부등호 방향이
바뀌는 지점이 context.py/caption.py 전체에 퍼져 있는데, 여기서 변환
자체가 틀리면 그쪽 커밋에서 스냅샷 diff는 나겠지만 "어디가 틀렸는지"는
안 알려준다. 그래서 변환만 따로 먼저 잠근다.

sample_doc(세션 스코프 픽스처)을 재사용 — DoclingParser().parse(path)를
또 부르면 같은 프로세스 안에서 두 번째 Docling 파싱이 일어나 메모리
누적 이슈를 재현하게 된다.
"""
from evidence_chunker.parser.docling import DoclingParser


def test_bbox_flip_preserves_visual_order(sample_doc):
    """TOPLEFT 변환 후에도 시각적 상하 관계가 보존되는가."""
    parsed = DoclingParser().from_doc(sample_doc)

    assert parsed.texts, "빈 texts면 이 테스트가 아무것도 검증 못 함"

    for raw_item, blk in zip(sample_doc.texts, parsed.texts):
        d = raw_item.model_dump()
        prov = d.get("prov", [])
        if not prov:
            continue
        raw_bbox = prov[0].get("bbox", {})
        page_h = parsed.page_sizes[blk.page_no][1]

        assert blk.bbox.l == raw_bbox.get("l", 0.0)
        assert blk.bbox.r == raw_bbox.get("r", 0.0)
        assert blk.bbox.t == page_h - raw_bbox.get("t", 0.0)
        assert blk.bbox.b == page_h - raw_bbox.get("b", 0.0)
        # BBox 불변식: TOPLEFT에서는 위쪽 모서리(t)가 아래쪽 모서리(b)보다 작아야 함
        assert blk.bbox.t < blk.bbox.b


def test_table_bbox_flip(sample_doc):
    """표 bbox도 텍스트와 같은 _to_bbox()를 타지만, 이 프로젝트의 핵심 대상이니
    별도로 확인해둔다."""
    parsed = DoclingParser().from_doc(sample_doc)
    assert parsed.tables, "빈 tables면 이 테스트가 아무것도 검증 못 함"

    for raw_table, blk in zip(sample_doc.tables, parsed.tables):
        d = raw_table.model_dump()
        prov = d.get("prov", [])
        if not prov:
            continue
        raw_bbox = prov[0].get("bbox", {})
        page_h = parsed.page_sizes[blk.page_no][1]

        assert blk.bbox.t == page_h - raw_bbox.get("t", 0.0)
        assert blk.bbox.b == page_h - raw_bbox.get("b", 0.0)
        assert blk.bbox.t < blk.bbox.b


def test_bbox_flip_same_relative_order_as_raw(sample_doc):
    """원본에서 A가 B보다 위(BOTTOMLEFT: cy_A > cy_B)였다면,
    변환 후에도 A가 B보다 위(TOPLEFT: cy_A < cy_B)여야 한다."""
    parsed = DoclingParser().from_doc(sample_doc)

    same_page = {}
    for raw_item, blk in zip(sample_doc.texts, parsed.texts):
        same_page.setdefault(blk.page_no, []).append((raw_item, blk))

    checked = 0
    for page_no, pairs in same_page.items():
        if len(pairs) < 2:
            continue
        (raw_a, blk_a), (raw_b, blk_b) = pairs[0], pairs[1]

        def _raw_cy(raw_item):
            prov = raw_item.model_dump().get("prov", [])
            if not prov:
                return None
            bbox = prov[0].get("bbox", {})
            return (bbox.get("t", 0.0) + bbox.get("b", 0.0)) / 2.0

        def _new_cy(blk):
            return (blk.bbox.t + blk.bbox.b) / 2.0

        raw_cy_a, raw_cy_b = _raw_cy(raw_a), _raw_cy(raw_b)
        if raw_cy_a is None or raw_cy_b is None or raw_cy_a == raw_cy_b:
            continue

        raw_a_above = raw_cy_a > raw_cy_b  # BOTTOMLEFT: 큰 y = 위
        new_a_above = _new_cy(blk_a) < _new_cy(blk_b)  # TOPLEFT: 작은 y = 위
        assert raw_a_above == new_a_above
        checked += 1

    assert checked > 0, "같은 페이지에 텍스트가 2개 이상인 경우가 없어 순서 비교를 못 함"
