# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
알고리즘 4개 모듈(caption/context/filters/flatten)이 Docling 없이도
import된다는 것을 sys.meta_path 훅으로 docling import를 실제로 막고
확인한다. docling과 docling_core(DoclingDocument 타입을 정의하는 별도
패키지) 둘 다 막는다 — 하나만 막으면 누가 docling_core.types를 알고리즘
모듈에 직접 쓰는 결합을 추가해도 이 테스트가 못 잡는다.

표 추출 알고리즘이 특정 파서에 묶여 있지 않다는 것(= 파서를 갈아끼울 수
있다는 것)을 이 테스트의 통과 자체가 보증한다.
evidence_chunker.chunker/parser.docling은 여기 포함되지 않는다 — 그쪽은
DocumentConverter/HybridChunker를 실제로 쓰는 오케스트레이션 레이어라
Docling이 당연히 필요하다.

pip uninstall docling으로 직접 확인하는 대신 import 훅으로 시뮬레이션한다
— 이 저장소의 다른 테스트/개발 워크플로가 docling 설치를 전제하므로,
CI에서까지 안전하게 반복 가능한 형태로 남겨둔다.
"""
import importlib
import sys

import pytest

_ALGORITHM_MODULES = [
    "evidence_chunker.caption",
    "evidence_chunker.context",
    "evidence_chunker.filters",
    "evidence_chunker.flatten",
    "evidence_chunker.parser.base",
]


_BLOCKED_PREFIXES = ("docling", "docling_core")  # docling-core는 별도 패키지(DoclingDocument 타입 정의)


class _BlockDocling:
    def find_spec(self, name, path=None, target=None):
        if name in _BLOCKED_PREFIXES or name.startswith(tuple(p + "." for p in _BLOCKED_PREFIXES)):
            raise ImportError(f"{name} blocked for test")
        return None


@pytest.mark.parametrize("module_name", _ALGORITHM_MODULES)
def test_algorithm_module_imports_without_docling(module_name):
    blocker = _BlockDocling()
    sys.meta_path.insert(0, blocker)
    # 이미 import된 상태면 훅이 안 타므로, 모듈 캐시에서 지우고 다시 import.
    saved = {
        name: mod for name, mod in list(sys.modules.items())
        if name == module_name or name.startswith(module_name + ".")
    }
    for name in saved:
        del sys.modules[name]

    try:
        importlib.import_module(module_name)
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
