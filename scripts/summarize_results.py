"""
summarize_results.py — results/bench_v6_*.json 을 읽어서 GitHub Actions 의
Step Summary(Actions 탭에서 바로 보이는 마크다운)로 출력한다.

benchmark.py의 summary 스키마(overall 없음, page_hit_at_k/evidence_hit_at_1/
table_hit_at_k/by_type 등)를 기준으로 한다 — measure_recall.py가 옛 스키마를
쓰던 시절 이 스크립트도 옛 필드명(overall/baseline_recall/eu_recall)을
읽었는데, 엔진이 benchmark.py로 통합되며 같이 옮겨졌다.

사용법 (CI 안에서):
    python scripts/summarize_results.py --label "dev-only (20개)"
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--results-dir", default="results", help="bench_v6_*.json 이 있는 디렉토리")
    p.add_argument("--label", default="", help="이번 실행 범위 라벨 (smoke/dev/full 등)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "bench_*.json")))
    files = [f for f in files if not f.endswith("_rows.json")]
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")

    lines = ["# 벤치마크 결과\n"]
    if args.label:
        lines.append(f"실행 범위: **{args.label}**\n")

    if not files:
        lines.append("결과 파일을 찾지 못했습니다 (실행이 중간에 실패했을 수 있어요).\n")

    for f in files:
        data = json.load(open(f, encoding="utf-8"))
        n = data.get("n_questions", "-")
        page_hit_1 = (data.get("page_hit_at_k") or {}).get("1") or (data.get("page_hit_at_k") or {}).get(1) or {}
        evidence_hit = data.get("evidence_hit_at_1") or {}

        lines.append(f"\n## {os.path.basename(f)}\n")
        lines.append("| 지표 | baseline | EU | 문항 수 |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| PageHit@1 | {page_hit_1.get('base', '-')} | "
            f"{page_hit_1.get('eu', '-')} | {n} |"
        )
        lines.append(
            f"| evidence_hit@1 | {evidence_hit.get('base', '-')} | "
            f"{evidence_hit.get('eu', '-')} | {n} |"
        )

        table_hit_1 = None
        if data.get("table_hit_at_k"):
            table_hit_1 = (data["table_hit_at_k"].get("1") or data["table_hit_at_k"].get(1))
        if table_hit_1:
            lines.append(
                f"| TableHit@1 | {table_hit_1.get('base', '-')} | "
                f"{table_hit_1.get('eu', '-')} | {data.get('n_table_questions', '-')} |"
            )

        stats = data.get("paired_statistics_page_hit_1_base_vs_eu")
        if stats:
            ci = stats.get("cluster_bootstrap_ci95_pp")
            ci_str = f"[{ci[0]:+.1f}, {ci[1]:+.1f}]pp" if ci else "N/A"
            lines.append(
                f"\nPageHit@1 문서단위 부트스트랩 차이: **{stats['difference_pp']:+.1f}pp** "
                f"(95% CI {ci_str})\n"
            )

        by_type = data.get("by_type") or {}
        type_blocks = {t: b for t, b in by_type.items() if b}
        if type_blocks:
            lines.append("\n**유형별 evidence_hit (baseline → EU)**\n")
            lines.append("| 유형 | baseline | EU | n |")
            lines.append("|---|---|---|---|")
            for t, blk in type_blocks.items():
                lines.append(
                    f"| {t} | {blk.get('baseline_evidence_hit', '-')} | "
                    f"{blk.get('eu_evidence_hit', '-')} | {blk.get('n', '-')} |"
                )

    text = "\n".join(lines) + "\n"
    print(text)
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
