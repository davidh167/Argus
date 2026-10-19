#!/usr/bin/env python3
"""Fetches the pinned embedding model (nomic-embed-text v1.5, GGUF F16, Apache-2.0) and verifies it.

The file is Ollama's public `nomic-embed-text` model blob; Ollama names blobs by the SHA-256 of their
contents, so this digest pins the exact weights, and the same file is what libargus's agreement numbers
(reports/ollama_agreement.md) were measured with. Nothing is trusted until the hash matches.

  python3 tools/fetch_model.py [--dest DIR] [--verify-only]
Prints the final model path, which is what ARGUS_MODEL_PATH should be set to.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

DIGEST = "970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
SIZE = 274290656
URL = f"https://registry.ollama.ai/v2/library/nomic-embed-text/blobs/sha256:{DIGEST}"
DEFAULT_DEST = Path.home() / ".local/share/job-search-tool/models"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest_dir: Path, url: str = URL, digest: str = DIGEST) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / "nomic-embed-text-v1.5.f16.gguf"
    if final.exists() and sha256(final) == digest:
        return final
    part = final.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, part.open("wb") as out:  # noqa: S310 - fixed https URL
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    if sha256(part) != digest:
        part.unlink()
        raise SystemExit(f"checksum mismatch for {url}: refusing to keep the download")
    part.replace(final)
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument(
        "--verify-only", action="store_true", help="hash an existing file instead of downloading"
    )
    args = ap.parse_args()
    if args.verify_only:
        existing = args.dest / "nomic-embed-text-v1.5.f16.gguf"
        ok = existing.exists() and sha256(existing) == DIGEST
        print(f"{existing}: {'OK' if ok else 'missing or wrong hash'}")
        sys.exit(0 if ok else 1)
    print(fetch(args.dest))
