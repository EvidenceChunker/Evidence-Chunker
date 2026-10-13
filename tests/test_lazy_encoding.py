# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
tokens.py의 tiktoken 인코더 로딩이 지연(lazy)되는지 확인.

tiktoken.get_encoding()은 로컬 캐시가 없으면 인코더 파일을 네트워크에서
받아온다. 이걸 모듈 import 시점에 실행하면 오프라인 + 캐시 없는 환경에서
`import evidence_chunker`(패키지 __init__.py가 chunker.py -> split.py ->
tokens.py를 끌고 옴) 자체가 실패한다. 네트워크가 있는 CI에서는 재현되지
않는 종류의 문제라, 지연 로딩을 테스트한다.
"""
import tiktoken


def test_import_does_not_call_get_encoding(monkeypatch):
    """get_encoding()을 막아도(오프라인+캐시 없음 시뮬레이션) import는 성공해야 함.

    sys.modules에서 지웠다가 다시 import하는 동안 evidence_chunker.* 모듈
    객체가 전부 새로 생긴다. 끝나고 원래대로 복원하지 않으면, 이미 클래스/
    함수 참조를 들고 있는 다른 테스트 파일과 monkeypatch 대상 모듈이 서로
    다른 객체가 되어 엉뚱하게 깨진다(test_pipeline_parity.py에서 실제로
    발생). 그래서 test_no_docling_dependency.py와 같은 저장/복원 패턴을 쓴다.
    """
    def _blocked(*args, **kwargs):
        raise RuntimeError("get_encoding()이 import 시점에 불렸음 — 지연 로딩 깨짐")

    monkeypatch.setattr(tiktoken, "get_encoding", _blocked)

    import sys
    saved = {
        name: mod for name, mod in list(sys.modules.items())
        if name == "evidence_chunker" or name.startswith("evidence_chunker.")
    }
    for name in saved:
        del sys.modules[name]

    try:
        import evidence_chunker  # noqa: F401 — import 자체가 테스트 대상
    finally:
        for name in list(sys.modules):
            if name == "evidence_chunker" or name.startswith("evidence_chunker."):
                del sys.modules[name]
        sys.modules.update(saved)


def test_count_tokens_calls_get_encoding_lazily(monkeypatch):
    """실제 사용 시점에는 get_encoding()이 (그때 가서) 호출돼야 함."""
    from evidence_chunker import tokens

    tokens._ENCODING = None  # 이전 테스트/사용으로 캐시됐을 수 있으니 리셋
    called = []
    real_get_encoding = tiktoken.get_encoding

    def _spy(*args, **kwargs):
        called.append(args)
        return real_get_encoding(*args, **kwargs)

    monkeypatch.setattr(tiktoken, "get_encoding", _spy)

    tokens.count_tokens("hello world")

    assert called, "count_tokens() 호출 시점에 get_encoding()이 안 불림"
