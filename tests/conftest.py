# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
세션 전체에서 테스트 PDF 파싱을 정확히 한 번만 수행.

같은 프로세스 안에서 Docling 파싱을 반복하면 메모리가 누적돼
std::bad_alloc으로 죽는 경우가 있다. 이 픽스처가 없으면 파싱이 필요한
테스트마다 개별적으로 변환을 돌려 그 상황을 반복 재현하게 된다.
"""
import os

import pytest

SAMPLE_PDF = os.path.join(
    os.path.dirname(__file__), "..", "data", "pdfs", "docling_technical_report.pdf"
)


@pytest.fixture(scope="session")
def sample_doc():
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("테스트 PDF 없음 (benchmarks/tools/download_test_pdfs.py로 받을 것)")

    from docling.document_converter import DocumentConverter

    return DocumentConverter().convert(SAMPLE_PDF).document
