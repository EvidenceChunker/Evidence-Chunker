"""
summarize_results.py — results/bench_*.json 을 읽어서 GitHub Actions 의
Step Summary(Actions 탭에서 바로 보이는 마크다운)로 출력한다.

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
    p.add_argument("--results-dir", default="results", help="bench_*.json 이 있는 디렉토리")
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
        overall = data.get("overall") or {}
        lines.append(f"\n## {os.path.basename(f)}\n")
        lines.append("| 지표 | baseline | EU | 문항 수 |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| Recall | {overall.get('baseline_recall', '-')} | "
            f"{overall.get('eu_recall', '-')} | {overall.get('n', '-')} |"
        )
        lines.append(
            f"| EM | {overall.get('baseline_em', '-')} | "
            f"{overall.get('eu_em', '-')} | {overall.get('n', '-')} |"
        )

        by_type = data.get("by_type") or {}
        type_blocks = {t: b for t, b in by_type.items() if b}
        if type_blocks:
            lines.append("\n**유형별 EM (baseline → EU)**\n")
            lines.append("| 유형 | baseline EM | EU EM | n |")
            lines.append("|---|---|---|---|")
            for t, blk in type_blocks.items():
                lines.append(
                    f"| {t} | {blk.get('baseline_em', '-')} | "
                    f"{blk.get('eu_em', '-')} | {blk.get('n', '-')} |"
                )

    text = "\n".join(lines) + "\n"
    print(text)
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
