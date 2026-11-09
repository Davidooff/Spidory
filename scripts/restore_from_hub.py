"""Download one machine's backup from the Hub and check every file against the Hub's listing.

Used by the rental-end script when copying from the machine fails, for example because the
machine already ended itself after a verified backup. Exits 0 only if every file on the Hub is
present locally with the same size, and with the same SHA-256 for files stored in LFS.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    api, out = HfApi(), Path(args.out)
    entries = [e for e in api.list_repo_tree(args.repo, path_in_repo=args.machine, recursive=True)
               if hasattr(e, "size")]
    if not any(e.path.endswith("DONE") or e.path.endswith("STOPPED") or e.path.endswith("LIMIT_REACHED")
               for e in entries):
        print("the Hub backup has no end marker; not trusting it as complete")
        return 1
    problems = []
    for entry in entries:
        relative = entry.path[len(args.machine) + 1:]
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        cached = Path(hf_hub_download(args.repo, entry.path))
        target.write_bytes(cached.read_bytes())
        lfs = getattr(entry, "lfs", None)
        digest = getattr(lfs, "sha256", None) if lfs is not None and not isinstance(lfs, dict) else (
            (lfs or {}).get("sha256") if isinstance(lfs, dict) else None)
        if target.stat().st_size != entry.size or (digest and sha256(target) != digest):
            problems.append(relative)
    print(f"restored {len(entries)} files, {len(problems)} differ: {problems[:5]}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
