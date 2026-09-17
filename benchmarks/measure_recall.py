"""
measure_recall.py — CI 벤치마크 러너. 평가 로직의 단일 소스는 benchmark.py이며,
이 파일은 benchmark.run()을 호출하고 DagsHub/MLflow 로깅만 담당하는 얇은
래퍼다. 채점 로직을 고치려면 benchmark.py만 고치면 된다.

사용법(GitHub Actions에서 실제로 쓰는 형태):
    python benchmarks/measure_recall.py \\
        --pdf-dir ./data/pdfs --qa-dir ./data/auto_qa --out-dir ./results \\
        --mlflow --dev-only [--max-pdfs 5]

    python measure_recall.py --pdf-dir ./data/pdfs --qa-dir ./auto_qa --out-dir ./results
    python measure_recall.py --pdf-dir ./data/pdfs --qa-dir ./auto_qa --out-dir ./results --dev-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# benchmark.py는 같은 디렉토리에 있다. 다른 위치에서 -m 등으로 실행될 경우까지 대비.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bv6  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--pdf-dir", type=Path, required=True, help="벤치마크 PDF 디렉토리")
    p.add_argument("--qa-dir", type=Path, required=True,
                   help="generate_qa_docling.py가 만든 {문서}_qa.json이 있는 디렉토리")
    p.add_argument("--out-dir", type=Path, required=True, help="결과 JSON을 저장할 디렉토리")
    p.add_argument("--dev-only", action="store_true", help="dev 서브셋(20개 문서)만 실행")
    p.add_argument("--max-pdfs", type=int, default=None, help="추가 상한 (디버깅/smoke용)")
    p.add_argument("--mlflow", action="store_true", help="DagsHub/MLflow에 결과 로깅")
    p.add_argument("--gemini-api-key", type=str, default=None,
                   help="LLM 실답변 평가용. 미지정 시 GEMINI_API_KEY 환경변수 사용, "
                        "둘 다 없으면 그 단계만 건너뜀(CI 기본 상태)")
    p.add_argument("--llm-sample-n", type=int, default=bv6.LLM_SAMPLE_N,
                   help="LLM 실답변 평가 샘플 문항 수(유형별 층화)")
    p.add_argument("--bbox-threshold", type=float, default=None, help="BBOX_THRESHOLD")
    p.add_argument("--sim-threshold", type=float, default=None, help="SIM_THRESHOLD")
    p.add_argument("--embed-model", type=str, default=None, help="EMBED_MODEL_NAME")
    p.add_argument("--encode-batch", type=int, default=None, help="ENCODE_BATCH")
    p.add_argument("--context-token-budget", type=int, default=None, help="CONTEXT_TOKEN_BUDGET")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.bbox_threshold is not None:
        bv6.BBOX_THRESHOLD = args.bbox_threshold
    if args.sim_threshold is not None:
        bv6.SIM_THRESHOLD = args.sim_threshold
    if args.embed_model is not None:
        bv6.EMBED_MODEL_NAME = args.embed_model
    if args.encode_batch is not None:
        bv6.ENCODE_BATCH = args.encode_batch
    if args.context_token_budget is not None:
        bv6.CONTEXT_TOKEN_BUDGET = args.context_token_budget

    gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    tag = "dev20" if args.dev_only else "full90"

    mlflow = None
    if args.mlflow:
        import mlflow as _mlflow
        mlflow = _mlflow
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow_db = (Path.cwd() / "mlflow.db").resolve()
            mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
        mlflow.set_experiment("evidence-chunker-benchmark")
        mlflow.start_run(run_name=tag)
        mlflow.log_params({
            "tag": tag,
            "bbox_threshold": bv6.BBOX_THRESHOLD,
            "sim_threshold": bv6.SIM_THRESHOLD,
            "embed_model": bv6.EMBED_MODEL_NAME,
            "encode_batch": bv6.ENCODE_BATCH,
            "max_pdfs": args.max_pdfs,
            "engine": "benchmark.py",
        })

    try:
        summary = bv6.run(args.pdf_dir, args.qa_dir, args.out_dir, args.dev_only, args.max_pdfs,
                           gemini_api_key, args.llm_sample_n)
    except Exception:
        if mlflow is not None:
            mlflow.end_run(status="FAILED")
        raise

    if mlflow is not None:
        # arm별 dict를 "지표_arm" 형태로 평탄화해서 MLflow에 로깅
        flat = {}
        for k, arms in summary["page_hit_at_k"].items():
            flat.update({f"page_hit_at_{k}_{a}": v for a, v in arms.items()})
        flat.update({f"evidence_hit_at_1_{a}": v for a, v in summary["evidence_hit_at_1"].items()})
        if summary.get("table_hit_at_k"):
            for k, arms in summary["table_hit_at_k"].items():
                flat.update({f"table_hit_at_{k}_{a}": v for a, v in arms.items()})
        flat["n_questions"] = summary["n_questions"]
        flat["n_table_questions"] = summary["n_table_questions"]
        if summary.get("paired_statistics_page_hit_1_base_vs_eu"):
            flat["page_hit_1_diff_pp"] = summary["paired_statistics_page_hit_1_base_vs_eu"]["difference_pp"]
        if summary.get("llm_answer_eval", {}).get("summary"):
            llm = summary["llm_answer_eval"]["summary"]
            flat["llm_baseline"] = llm["baseline"]
            flat["llm_eu"] = llm["treatment"]
            flat["llm_diff_pp"] = llm["difference_pp"]
        for t, blk in summary.get("by_type", {}).items():
            if not blk:
                continue
            flat.update({f"type_{t}_{k}": v for k, v in blk.items() if isinstance(v, (int, float))})

        mlflow.log_metrics(flat)
        mlflow.log_artifact(str(args.out_dir / f"bench_v6_{tag}.json"))
        mlflow.log_artifact(str(args.out_dir / f"bench_v6_{tag}_rows.json"))
        mlflow.end_run()


if __name__ == "__main__":
    main()
