<div align="center">
    
# Evidence-Chunker

**표와 주변 문맥을 연결하는 PDF 기반 RAG 청킹 라이브러리**

Docling으로 추출한 표·캡션·주변 설명을 `EvidenceUnit`으로 구성합니다.

[Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)

![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

[Quick Start](quick-start) · [Documentation](https://github.com/#README/Evidence-Chunker/wiki) · [API Reference](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/API-Reference) · [Benchmark](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/Benchmark)

</div>


## Overview

PDF에서 표를 추출해도, 표의 값만으로는 질문에 답하기 어려울 수 있습니다. 표에 연결된 캡션, 본문 설명, 각주가 별도 청크로 분리되면 검색된 근거만으로 의미를 파악하기 어렵기 때문입니다.

Evidence-Chunker는 표를 중심으로 관련 정보를 모아 하나의 **EvidenceUnit**을 생성합니다. 표를 포함한 문서 구조 분석은 기본적으로 [Docling](https://github.com/docling-project/docling)에 맡기고, 그 결과를 활용해 **표와 문맥의 연결 관계**를 구성합니다.

```
PDF ──> Docling Parser ──> ParsedDoc ──> Evidence-Chunker
                                             │
                                             ▼
                                        EvidenceUnit
                                       /            \
                         retrieval_units             text
                         (small search inputs)       (full evidence)
```

검색에서는 행, 표 요약, 문맥 등의 작은 표현을 활용하고, 검색된 표현이 속한 부모 EvidenceUnit에서 캡션, 표, 관련 문맥을 함께 가져올 수 있습니다.

## Features

| 기능 | 설명 |
| --- | --- |
| **Table-aware evidence** | 표, 캡션, 섹션 제목, 관련 설명 문단, 각주 및 출처 정보를 연결 |
| **Caption matching** | 명시적 캡션 참조를 우선 사용하고 위치 기반 보완 탐색 수행 |
| **Context attachment** | 표 주변 문단을 거리·섹션 등 조건으로 선택 (기본 `300pt`) |
| **Search / context separation** | `retrieval_units`(작은 검색 표현)와 `text`(전체 근거)를 분리 |
| **Large-table handling** | 크기가 큰 표에 행 단위 분할 정책 적용 (모든 표에서 최대 토큰 길이를 보장하지 않음) |
| **Parser adapters** | `ParsedDoc` 계약을 만족하는 커스텀 파서로 표 기반 `chunk()` 실행 가능 |
| **RAG integrations** | LangChain `Document`, LlamaIndex `TextNode` 변환 지원 |
| **Local evidence construction** | 기본 근거 구성에 LLM API 호출이나 임베딩 모델이 필요하지 않음 |

## Installation

Python 3.10 이상이 필요합니다.

```bash
git clone https://github.com/EvidenceChunker/Evidence-Chunker.git
cd Evidence-Chunker
pip install -e .
```

프레임워크나 유사도 필터를 사용할 때만 해당 extra를 설치하세요.

```bash
pip install -e '.[langchain]'    # LangChain
pip install -e '.[llamaindex]'   # LlamaIndex
pip install -e '.[similarity]'   # 임베딩 기반 문맥 유사도 필터
```

처음 사용하는 환경에서는 Docling 모델 아티팩트 다운로드가 필요할 수 있습니다. 오프라인 실행을 위한 준비 사항은 [Configuration](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/Configuration)에 정리했습니다.

## Quick Start

### 1. EvidenceUnit 생성

```python
from evidence_chunker import EvidenceChunker

chunker = EvidenceChunker()
units = chunker.chunk("report.pdf")

for unit in units:
    print(unit.eu_id, unit.safe_caption)
    print(unit.text)
```

`chunk()`는 표 중심의 `EvidenceUnit` 목록을 반환합니다. 표가 없는 문서라면 빈 목록이 될 수 있습니다.

### 2. 검색 표현과 전체 근거 구분

```python
unit = units[0]  # EvidenceUnit이 생성된 문서에서 실행

print(unit.retrieval_units[:3])  # 임베딩 검색용 작은 텍스트들
print(unit.text)                 # 표·캡션·문맥이 포함된 전체 근거
```

`retrieval_units`는 별도의 검색 후보들이지만, 각각의 표현은 동일한 부모 EvidenceUnit에 연결됩니다. 실제 검색에는 임베딩 모델과 벡터스토어를 연결해야 합니다.

### 3. 일반 본문을 포함한 코퍼스 구성

```python
corpus = chunker.build_corpus("report.pdf")
# EvidenceUnit(표) + TextChunk(일반 본문)
```

`build_corpus()`에서 일반 본문 청킹은 Docling `HybridChunker`를 사용합니다. **커스텀 파서 주입은 `chunk()` 경로에서만 지원**하며, 커스텀 파서를 이용한 `build_corpus()`는 현재 지원하지 않습니다.

## LangChain / LlamaIndex

표현별로 검색한 뒤 부모 근거를 활용하는 *small-to-big* 인덱싱을 지원합니다.

**LangChain**

```python
from evidence_chunker.export.langchain import to_langchain_units

search_docs = to_langchain_units(corpus)

for doc in search_docs[:3]:
    search_text = doc.page_content
    full_evidence = doc.metadata.get("parent_text", search_text)
    print(search_text[:100])
    print(full_evidence[:200])
```

**LlamaIndex**

```python
from evidence_chunker.export.llamaindex import to_llamaindex_units

nodes = to_llamaindex_units(corpus)
```

LangChain/LlamaIndex의 `EvidenceRetriever` 래퍼와 중복 제거 기능도 제공합니다. 단, **중복 제거가 자동으로 전체 후보에 대한 max-pooling 검색을 수행하는 것은 아닙니다.** 사용 방식은 [Examples](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/Examples)와 [API Reference](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/API-Reference)를 참고하세요.

## Benchmark

Docling `HybridChunker`와 Evidence-Chunker **v0.1.2**를 동일한 조건에서 비교했습니다.

- **Dataset:** 90 PDF / 구조 검증된 QA 2,239개
- **Environment:** Docling 2.143.1 / Docling Core 2.105.0
- **Embedding:** `all-MiniLM-L6-v2`

| Metric | HybridChunker | Evidence-Chunker |
| --- | --- | --- |
| [PageHit@1](mailto:PageHit@1) | 58.91% | **62.26%** |
| [TableHit@1](mailto:TableHit@1) | 52.43% | **59.00%** |
| [**EvidenceHit@1**](mailto:EvidenceHit@1) | 31.22% | **49.44%** |

`EvidenceHit@1`은 상위 1개 근거에 **정답 값과 필요한 문맥 키워드가 함께 존재하는지** 검사한 지표입니다. **LLM 답변 정확도는 아닙니다.**

> 검색용 행 텍스트의 반복 캡션 prefix를 제거한 **실험 변형**에서는 [EvidenceHit@1이](mailto:EvidenceHit@1%EC%9D%B4) 64.14%였지만, 이 변경은 **v0.1.2 기본 구현에 포함되지 않습니다.**
> 

평가 조건, 문항 선정, Ablation 등 상세 결과는 [Benchmark](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/Benchmark)와 [Experiments](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/Experiments)를 참고하세요.

## Configuration

```python
chunker = EvidenceChunker(
    bbox_threshold=300.0,  # 주변 문맥 탐색 거리(pt)
    sim_threshold=0.0,     # 0이면 임베딩 기반 유사도 필터 비활성화
)
```

설정 전체와 파라미터 설명은 [Configuration](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/Configuration)에 있습니다.

## Limitations

- 표 추출 품질과 문서 레이아웃 인식은 기본 Docling 파서의 결과에 영향을 받습니다.
- 복합 헤더, 병합 셀, 여러 페이지에 걸친 표는 추가적인 처리 개선이 필요할 수 있습니다.
- 전체 EvidenceUnit을 LLM에 전달하면 일반 청크보다 **입력 토큰 수가 증가**할 수 있습니다.
- `chunk()`의 커스텀 파서 지원과 `build_corpus()`의 Docling 의존성은 구분해야 합니다.

자세한 설계 제약과 실험 결과는 Wiki를 참고하세요.

## Documentation

| Page | Contents |
| --- | --- |
| [Architecture](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/Architecture) | 내부 데이터 흐름과 EvidenceUnit 구조 |
| [API Reference](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/API-Reference) | 함수·클래스·반환값 |
| [Configuration](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/Configuration) | 기본값 및 선택적 설정 |
| [Examples](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/Examples) | Python·LangChain·LlamaIndex 예제 |
| [Benchmark](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/Benchmark) | 평가 환경·지표·전체 결과 |
| [Experiments](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/Experiments) | Ablation·설계 실험 |
| [QA Generation](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/QA-Generation) | QA 생성 및 검증 절차 |

## Development

```bash
pip install -e '.[dev]'
pytest -q
```

버그 신고 및 기여 제안은 [GitHub Issues](https://github.com/EvidenceChunker/Evidence-Chunker/issues)를 사용해주세요. 보안 취약점 신고는 [SECURITY.md](SECURITY.md)를 참고하세요.

## Acknowledgements

- [Docling](https://github.com/docling-project/docling) — PDF parsing 및 표 구조 추출
- [Han (2026), *Evidence Units: Ontology-Grounded Document Organization for Parser-Independent Retrieval*](https://arxiv.org/abs/2604.00500) — Evidence Unit의 상위 개념 참고. 논문의 전체 시스템을 구현한 것은 아닙니다.

## License

[Apache License 2.0](LICENSE). API 및 벤치마크 표는 **v0.1.2 ([commit `87a43e4`](https://github.com/EvidenceChunker/Evidence-Chunker/commit/87a43e4f15e50b3ad27e91769ffe9e1864e06791))** 기준입니다.
