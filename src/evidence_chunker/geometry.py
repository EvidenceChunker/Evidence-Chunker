# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
geometry.py

EvidenceUnit.bbox 좌표 변환.

이 패키지의 좌표계는 두 종류다. 파서 경계를 지난 내부 좌표
(parser.base.BBox)는 계산 편의를 위해 TOPLEFT로 정규화돼 있지만,
EvidenceUnit.bbox는 0~1로 스케일한 BOTTOMLEFT를 공개 계약으로 유지한다.

여기가 그 마지막 변환을 담당하는 유일한 지점 — 변환 규칙은
normalize_bbox() 참고.
"""

from __future__ import annotations


def normalize_bbox(
    bbox: dict,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    """
    Docling PDF 포인트 bbox (BOTTOMLEFT origin) → 0~1 normalized (x1,y1,x2,y2).

    Docling bbox 키: l(left), t(top), r(right), b(bottom), coord_origin=BOTTOMLEFT
    BOTTOMLEFT에서 t > b (top이 y축 더 큼).

    반환값은 BOTTOMLEFT 유지, 0~1로만 스케일 축소:
        x1 = l / width
        y1 = b / height   (시각적 하단, 작은 y)
        x2 = r / width
        y2 = t / height   (시각적 상단, 큰 y)
    """
    l = bbox.get("l", 0.0)
    t = bbox.get("t", 0.0)
    r = bbox.get("r", 0.0)
    b = bbox.get("b", 0.0)
    return (
        l / page_width,
        b / page_height,
        r / page_width,
        t / page_height,
    )
