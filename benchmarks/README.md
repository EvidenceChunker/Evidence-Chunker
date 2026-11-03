# Benchmark & Reproducibility

Evidence-Chunker v0.1.2와 Docling `HybridChunker`의 검색 근거 보존 성능을 동일한 PDF, QA, 파싱 결과 및 임베딩 모델로 비교한 평가 코드와 설정입니다.

> **평가 범위:** 검색된 근거의 포함 여부를 측정합니다. LLM이 생성한 답변의 정확도나 서비스의 end-to-end latency를 측정한 실험은 아닙니다.

## Dataset

| 항목 | 값 |
| --- | ---: |
| 평가 문서 | 90 PDF |
| 최초 자동 생성 QA | 2,725 |
| 독립 감사 후 사용한 QA | 2,239 |
| 감사 단계에서 보류한 QA | 486 |
| 현재 Docling 테이블 인덱스로 재매핑한 QA | 602 |
| 평가 처리 실패 | 0 PDF |

QA 생성과 감사 과정에서는 평가 대상 청커를 사용하지 않습니다. 감사 결과로 선정된 QA를 **두 청커에 동일하게** 적용합니다. 감사는 원본 파싱 결과의 구조적 일관성을 확인하는 절차이며, 사람의 정답 주석 검증과는 다릅니다.

실제 평가에 사용한 PDF 파일명은 `results/pdf_audit_summary.json`의 `doc_id`에서 확인할 수 있습니다. PDF 원본과 원본 QA는 이 디렉터리에 포함되지 않으며, 재실행 시 감사에 기록된 SHA-256과 일치해야 합니다.

## Repository Layout

```text
benchmarks/
├── README.md
├── qa/
│   ├── generate_qa_docling.py       # QA 자동 생성
│   ├── qa_source_integrity_audit.py # 원본 QA 독립 감사
│   └── build_verified_qa.py        # 감사 승인 QA 재구성
├── evaluation/
│   ├── paired_benchmark.py         # 검색 평가, Ablation, 부트스트랩
│   └── token_budget_replay.py      # Top-k 근거 및 토큰 예산 재평가
├── configs/
│   ├── parser_config.json
│   ├── run_config.json
│   └── environment_manifest.json
└── results/
    ├── qa_audit_summary.json
    ├── pdf_audit_summary.json
    ├── verified_qa_mapping.json
    └── benchmark_summary.json
```

`results/benchmark_summary.json`은 토큰 예산 재평가 실행의 최종 결과이며, 기본 검색 지표 및 Ablation 결과도 포함합니다.

## Environment

| 구성 요소 | 고정 버전 / 조건 |
| --- | --- |
| Python | 3.13.15 (기록된 실행 환경) |
| Docling | 2.143.1 |
| Docling Core | 2.105.0 |
| Evidence-Chunker | 0.1.2, commit `87a43e4f15e50b3ad27e91769ffe9e1864e06791` |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Sentence Transformers | 5.7.0 |
| GPU (기록된 실행) | Tesla T4 |
| Evidence-Chunker 설정 | `bbox=300.0`, `sim=0.0` |
| 통계 | PDF 단위 군집 부트스트랩 10,000회, seed `20261029` |

`configs/parser_config.json`에는 QA 감사에 사용한 Docling 파싱 옵션이 저장되어 있습니다. 평가 실행 코드는 현재 파싱 옵션이 감사 설정과 일치하는지 검사합니다. 전체 환경 정보는 `configs/environment_manifest.json`과 `configs/run_config.json`을 참고하세요.

## Reproduce

아래 예제는 **저장소 루트**에서 실행하는 명령어입니다. 입력 파일은 다음과 같이 준비합니다.

- `PDF_DIR`: 원본 90개 PDF. 파일명은 `<doc_id>.pdf` 형식이어야 합니다.
- `ORIGINAL_QA_DIR`: 최초 2,725개 QA가 담긴 `*_qa.json` 파일 90개.
- `VERIFIED_QA_DIR`: `build_verified_qa.py`로 생성할 2,239개 검증 QA 디렉터리.
- `PINNED_REPO`: 평가 당시 커밋으로 체크아웃한 **별도의** Evidence-Chunker 저장소.

```bash
export PDF_DIR=/path/to/pdfs
export ORIGINAL_QA_DIR=/path/to/original_qa
export VERIFIED_QA_DIR=/path/to/QA_v2_verified
export PINNED_REPO=/path/to/Evidence-Chunker-pinned
export MODEL_SHA=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

### 1. Prepare the pinned implementation

```bash
git clone https://github.com/EvidenceChunker/Evidence-Chunker.git "$PINNED_REPO"
git -C "$PINNED_REPO" checkout --detach 87a43e4f15e50b3ad27e91769ffe9e1864e06791
```

Docling 및 관련 의존성을 위 표에 기재된 버전으로 설치해야 합니다. 토큰 효율 평가에는 `tiktoken`도 필요합니다. 외부 모델이 처음 다운로드되는 환경에서는 네트워크 또는 사전 캐시가 필요합니다.

### 2. Generate and audit QA

원본 QA가 없다면 생성기를 사용할 수 있습니다. 단, 이미 공개된 QA 데이터셋을 정확히 재현하려면 생성된 파일의 SHA-256이 감사 매니페스트와 일치해야 합니다.

```bash
python benchmarks/qa/generate_qa_docling.py \
  --pdf-dir "$PDF_DIR" \
  --out-dir "$ORIGINAL_QA_DIR"

python benchmarks/qa/qa_source_integrity_audit.py \
  --pdf-dir "$PDF_DIR" \
  --qa-dir "$ORIGINAL_QA_DIR" \
  --out ./benchmark_runs/audit \
  --max-pdfs 90 \
  --docling-version 2.143.1 \
  --docling-core-version 2.105.0 \
  --evidence-sha 87a43e4f15e50b3ad27e91769ffe9e1864e06791
```

감사 스크립트는 문항별 검사 결과와 테이블 매핑 후보를 출력합니다. **고정된 최종 QA 선정 목록**은 `results/verified_qa_mapping.json`에 보관되어 있으며, 이 파일을 감사 스크립트가 자동으로 다시 생성하지는 않습니다.

### 3. Build verified QA

```bash
python benchmarks/qa/build_verified_qa.py \
  --qa-dir "$ORIGINAL_QA_DIR" \
  --bundle-dir benchmarks/results \
  --out-dir "$VERIFIED_QA_DIR"
```

이 단계는 감사 당시 원본 QA의 SHA-256을 검사한 뒤 승인 문항만 복사하고, `meta.table_index`를 현재 파싱 결과에 맞춰 재매핑합니다. `VERIFIED_QA_DIR`은 존재하지 않거나 비어 있어야 합니다.

### 4. Run paired retrieval benchmark

```bash
python benchmarks/evaluation/paired_benchmark.py \
  --pdf-dir "$PDF_DIR" \
  --qa-dir "$VERIFIED_QA_DIR" \
  --audit-dir benchmarks/results \
  --repo-dir "$PINNED_REPO" \
  --model-sha "$MODEL_SHA" \
  --out-dir ./benchmark_runs/paired \
  --max-pdfs 90 --bbox 300 --sim 0 --batch-size 64
```

### 5. Run token-budget replay

```bash
python benchmarks/evaluation/token_budget_replay.py \
  --pdf-dir "$PDF_DIR" \
  --qa-dir "$VERIFIED_QA_DIR" \
  --audit-dir benchmarks/results \
  --repo-dir "$PINNED_REPO" \
  --model-sha "$MODEL_SHA" \
  --out-dir ./benchmark_runs/token_budget \
  --max-pdfs 90 --bbox 300 --sim 0 --batch-size 64
```

**주의:** 두 평가 스크립트의 `--max-pdfs` 기본값은 `3`입니다. 90개 전체 평가에는 `--max-pdfs 90`을 명시해야 합니다. `--repo-dir`은 벤치마크 코드를 추가한 개발 브랜치가 아니라 고정 커밋의 별도 체크아웃을 가리켜야 합니다.

## Metrics

- **PageHit@k:** 상위 `k`개 근거에 정답이 위치한 PDF 페이지가 포함되는지 확인합니다.
- **TableHit@k:** 상위 `k`개 근거에 정답 표의 인덱스가 포함되는지 확인합니다.
- **EvidenceHit@1:** 검색 결과 1위의 **반환 근거 텍스트**에 `answer_spec.value`와 모든 `context_keys`가 함께 포함되는지 확인합니다.

HybridChunker와 Evidence-Chunker는 같은 Docling 파싱 결과를 사용합니다. Evidence-Chunker에서는 여러 작은 `retrieval_units`의 검색 점수를 부모 EvidenceUnit으로 집계하며, 최종 근거는 부모 단위로 평가합니다. 이 지표는 문자열 포함 기반 검색 평가로, 생성 답변의 Exact Match가 아닙니다.

## Reference Results

90 PDF / 2,239 QA의 기록된 결과입니다.

| Metric | HybridChunker | Evidence-Chunker v0.1.2 | Difference |
| --- | ---: | ---: | ---: |
| PageHit@1 | 58.91% | 62.26% | +3.35%p |
| TableHit@1 | 52.43% | 59.00% | +6.57%p |
| EvidenceHit@1 | 31.22% | 49.44% | **+18.22%p** |

EvidenceHit@1 차이의 95% CI는 PDF 단위 군집 부트스트랩으로 계산한 **[+13.02, +23.79]%p**입니다. PageHit@1 차이의 CI는 **[−0.54, +7.44]%p**로 0을 포함합니다.

`eu_no_repeated_caption_prefix` 실험군의 EvidenceHit@1은 **64.14%**였습니다. 이는 **정식 v0.1.2 기능이 아닌 실험용 검색 표현 변형**이며, 동일 데이터셋에서 개선 방안을 검토한 탐색적 결과입니다.

Top-k, QA 유형별 수치, Ablation 및 토큰 예산 결과는 `results/benchmark_summary.json`과 [Benchmark Wiki](https://github.com/EvidenceChunker/Evidence-Chunker/wiki/Benchmark)를 참고하세요.

## Timing Notes

`benchmark_summary.json`에는 PDF별 `parse_s`, `hybrid_chunk_s`, `eu_chunk_s`, 임베딩 및 랭킹 시간이 기록되어 있습니다.

- Docling 파싱은 문서당 한 번 수행한 **공통 단계**입니다.
- `hybrid_chunk_s`와 `eu_chunk_s`는 동일 실행에서 **순차적으로 측정한 단계별 시간**입니다. `eu_chunk_s`에는 기존 HybridChunker 결과에서 일반 본문을 활용하는 작업도 포함되므로, 이를 두 시스템의 독립적인 전체 처리 시간 비교로 해석하면 안 됩니다.
- 모델 로딩, 패키지 설치 및 QA 생성은 측정에서 제외합니다.
- 토큰 예산 실험은 `cl100k_base`로 반환 문맥 길이를 측정한 프록시이며, 실제 LLM 비용이나 생성 품질을 측정하지 않습니다.

## Reproducibility & Limitations

- 이 디렉터리에는 **원본 PDF, 원본 QA, 검증된 QA 파일, 문항별 `rows_checkpoint.json`이 포함되지 않습니다.** 따라서 현재 파일만으로 전체 평가를 재실행하거나 문항별 지표를 독립 재계산할 수는 없습니다. 외부 재현을 위해서는 입력 데이터의 적법한 제공 경로와 문항별 결과를 별도로 공개해야 합니다.
- `verified_qa_mapping.json`은 QA 독립 감사의 승인 문항과 테이블 재매핑을 기록한 **고정 산출물**입니다. 문항 선정 과정을 완전히 자동 재생성하는 별도 스크립트는 현재 제공되지 않습니다.
- 소스 코드의 주석을 정리한 공개용 `.py`는 실행 로직이 같아도 파일 SHA-256이 달라집니다. `configs/environment_manifest.json`의 `benchmark_script_sha256`은 **당시 실행한 원본 토큰 재평가 스크립트의 해시**이며, 현재 `evaluation/token_budget_replay.py`의 해시가 아닙니다.
- 표·문맥 검색 개선은 확인되지만, 일부 QA 유형의 절대 성공률은 낮습니다. 모든 결과는 사용한 90개 PDF, 모델 및 설정의 범위에서 해석해야 합니다.
