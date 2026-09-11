"""
download_bench_data.py — DagsHub Storage 버킷(dnjsgkfka/Evidence-Chunker)에서
pdfs/, auto_qa/ 를 받아온다. CI(GitHub Actions)에서 Kaggle 대신 쓰는 용도.

환경변수 DAGSHUB_TOKEN 필요 (개인 액세스 토큰).

사용법:
    python scripts/download_bench_data.py --out-dir ./data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import dagshub
from dagshub.upload.wrapper import get_repo_bucket_client

REPO_FULL = "dnjsgkfka/Evidence-Chunker"
BUCKET_NAME = "Evidence-Chunker"  # 버킷 이름 = 레포 이름
PREFIXES = ["pdfs/", "auto_qa/"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--out-dir", type=Path, required=True, help="pdfs/, auto_qa/ 를 받을 루트 디렉토리")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        sys.exit("환경변수 DAGSHUB_TOKEN이 없습니다.")

    dagshub.auth.add_app_token(token)
    s3 = get_repo_bucket_client(REPO_FULL)

    total = 0
    for prefix in PREFIXES:
        local_dir = args.out_dir / prefix.rstrip("/")
        local_dir.mkdir(parents=True, exist_ok=True)

        continuation_token = None
        count = 0
        while True:
            kwargs = {"Bucket": BUCKET_NAME, "Prefix": prefix, "MaxKeys": 1000}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            resp = s3.list_objects_v2(**kwargs)

            for obj in resp.get("Contents", []):
                key = obj["Key"]
                fname = key[len(prefix):]
                if not fname:
                    continue
                dest = local_dir / fname
                s3.download_file(BUCKET_NAME, key, str(dest))
                count += 1

            if resp.get("IsTruncated"):
                continuation_token = resp.get("NextContinuationToken")
            else:
                break

        print(f"[{prefix}] {count}개 파일 다운로드 완료 -> {local_dir}")
        total += count

    print(f"\n총 {total}개 파일 다운로드 완료")


if __name__ == "__main__":
    main()
