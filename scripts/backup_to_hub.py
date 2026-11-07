"""Back up a machine's results to a private Hugging Face repo, then end its own rental.

Runs on the GPU machine. Every `interval` seconds it uploads logs, results and each run's latest
checkpoint into `<repo>/<machine>/`. When training has ended, it uploads once more, checks that
every local file is on the Hub with the same size, waits `grace` seconds so the laptop can copy
first, then destroys this instance with Vast's per-instance key. It never destroys before a
verified upload.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import time
from pathlib import Path

PATTERNS = ("*.json", "*.jsonl", "*.log", "*.txt", "*/last.pt", "DONE", "STOPPED", "LIMIT_REACHED")
ENDED = ("DONE", "STOPPED", "LIMIT_REACHED")
RUNNING = r"[r]un_v3.sh|[r]un_between.sh|[b]uild_then_between.sh|[s]pidory\.(train|rounds|evaluate|clean_v5)"


def wanted(runs: Path) -> list[Path]:
    """Files to back up: everything except numbered checkpoints and temporary files."""
    found = []
    for path in sorted(runs.rglob("*")):
        relative = path.relative_to(runs).as_posix()
        if path.is_file() and not relative.endswith(".tmp") and any(
                fnmatch.fnmatch(relative, p) or fnmatch.fnmatch(path.name, p) for p in PATTERNS):
            found.append(path)
    return found


def training_ended(runs: Path) -> bool:
    if any((runs / marker).exists() for marker in ENDED):
        return True
    busy = subprocess.run(["pgrep", "-f", RUNNING], capture_output=True).returncode == 0
    return not busy


def upload(api, repo: str, runs: Path, machine: str) -> int:
    files = wanted(runs)
    if not files:
        return 0
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(runs), path_in_repo=machine,
                      allow_patterns=[p.relative_to(runs).as_posix() for p in files],
                      commit_message=f"{machine}: backup of {len(files)} files")
    return len(files)


def verify(api, repo: str, runs: Path, machine: str) -> list[str]:
    """Local files missing on the Hub or with a different size. Empty means all backed up."""
    remote = {}
    for entry in api.list_repo_tree(repo, path_in_repo=machine, recursive=True, repo_type="model"):
        if hasattr(entry, "size"):
            remote[entry.path] = entry.size
    problems = []
    for path in wanted(runs):
        key = f"{machine}/{path.relative_to(runs).as_posix()}"
        if remote.get(key) != path.stat().st_size:
            problems.append(key)
    return problems


def destroy_self(log) -> bool:
    key, instance = os.environ.get("CONTAINER_API_KEY"), os.environ.get("CONTAINER_ID")
    if not key or not instance:
        environ = dict(line.split("=", 1) for line in
                       Path("/proc/1/environ").read_bytes().decode(errors="ignore").split("\0") if "=" in line)
        key, instance = environ.get("CONTAINER_API_KEY"), environ.get("CONTAINER_ID")
    if not key or not instance:
        log("no per-instance key found; the laptop script must end this rental")
        return False
    result = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-X", "DELETE",
                             "-H", f"Authorization: Bearer {key}",
                             f"https://console.vast.ai/api/v0/instances/{instance}/"], capture_output=True, text=True)
    log(f"destroy request answered {result.stdout}")
    return result.stdout.startswith("2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--grace", type=int, default=3600)
    parser.add_argument("--no-destroy", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi

    api, runs = HfApi(), Path(args.runs)
    log_path = runs / "backup_to_hub.log"

    def log(text: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {text}"
        print(line, flush=True)
        with open(log_path, "a") as sink:
            sink.write(line + "\n")

    while True:
        ended = training_ended(runs)
        try:
            log(f"uploaded {upload(api, args.repo, runs, args.machine)} files")
        except Exception as error:  # a failed upload is retried next time
            log(f"upload failed: {error}")
        if ended:
            break
        time.sleep(args.interval)

    for attempt in range(1, 7):
        try:
            upload(api, args.repo, runs, args.machine)
            problems = verify(api, args.repo, runs, args.machine)
        except Exception as error:
            problems = [f"error: {error}"]
        if not problems:
            log("final backup verified on the Hub")
            break
        log(f"verify attempt {attempt}: {len(problems)} files not backed up, e.g. {problems[:3]}")
        time.sleep(300)
    else:
        log("backup could not be verified: not destroying this machine")
        return
    if args.no_destroy:
        return
    log(f"waiting {args.grace} s so the laptop can copy first")
    time.sleep(args.grace)
    destroy_self(log)


if __name__ == "__main__":
    main()
