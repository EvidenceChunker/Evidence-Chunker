# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
tokens.py

EvidenceUnit 토큰 카운트 유틸리티 (tiktoken cl100k_base 기준).

DEFAULT_TOKEN_LIMIT은 EU 하나가 넘지 않아야 할 토큰 수이고, 이 한도를
실제로 소비해 표를 행 단위로 쪼개는 쪽은 split.py다. 인코더는 첫 카운트
시점까지 지연 로딩한다 — 이유는 _get_encoding() 참고.
"""

from .unit import EvidenceUnit

_ENCODING = None
DEFAULT_TOKEN_LIMIT = 512


def _get_encoding():
    """tiktoken.get_encoding()은 첫 호출 시 인코더 파일을 네트워크에서 받아올
    수 있어(로컬 캐시 없으면), 모듈 import 시점에 즉시 실행하면 오프라인
    환경에서 `import evidence_chunker`(패키지 __init__.py가 chunker.py를
    거쳐 이 모듈까지 끌고 옴) 자체가 실패한다. 실제로 첫 토큰 카운트가
    필요한 시점까지 지연."""
    global _ENCODING
    if _ENCODING is None:
        import tiktoken
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """텍스트의 토큰 수."""
    return len(_get_encoding().encode(text))


def count_eu_tokens(eu: EvidenceUnit) -> int:
    """EU 전체 텍스트(eu.text — table_html 포함)의 토큰 수."""
    return count_tokens(eu.text)


def exceeds_token_limit(eu: EvidenceUnit, limit: int = DEFAULT_TOKEN_LIMIT) -> bool:
    """EU가 한도를 넘겼는지 — split.py의 분할 대상 판정과 같은 기준."""
    return count_eu_tokens(eu) > limit


def get_token_status(eu: EvidenceUnit, limit: int = DEFAULT_TOKEN_LIMIT) -> dict:
    """EU 하나의 토큰 사용량 리포트(계측/디버깅용)."""
    token_count = count_eu_tokens(eu)
    return {
        "eu_id": eu.eu_id,
        "token_count": token_count,
        "limit": limit,
        "exceeds_limit": token_count > limit,
        "usage": f"{token_count/limit:.1%}",
    }
