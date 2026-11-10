"""Back up a machine's results to a private Hugging Face repo, then end its own rental.

Runs on the GPU machine. Logs and results go into `<repo>/<machine>/` every `interval` seconds,
and each run's latest checkpoint every `checkpoint_every` cycles. Training counts as ended only
when nothing is running and either an end marker exists, or training was seen running and then
two checks in a row find nothing (a crash). It then uploads everything, checks every file on the
Hub by SHA-256 (size for small files), waits a grace period, waits longer while the laptop says it
is copying, and destroys this instance with Vast's per-instance key, retrying until it succeeds.
It never destroys before a verified backup.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

PATTERNS = ("*.json", "*.jsonl", "*.log", "*.txt", "*/last.pt", "DONE", "STOPPED", "LIMIT_REACHED")
CHECKPOINTS = ("*/last.pt",)
ENDED = ("DONE", "STOPPED", "LIMIT_REACHED")
RUNNING = (r"[r]un_jobs.sh|[r]un_v3.sh|[r]un_between.sh|[b]uild_then_between.sh|[r]un_mixed.sh|[b]uild_mixed_then_run.sh|"
           r"[s]pidory\.(train|rounds|evaluate|clean_v5|grow_v4|mixed_data)")
LAPTOP_COPYING = "LAPTOP_COPYING"  # the laptop refreshes this file while it copies


def wanted(runs: Path, checkpoints: bool = True) -> list[Path]:
    """Files to back up: logs, results, markers and each run's latest checkpoint."""
    found = []
    for path in sorted(runs.rglob("*")):
        relative = path.relative_to(runs).as_posix()
        if not path.is_file() or relative.endswith(".tmp") or relative == LAPTOP_COPYING:
            continue
        match = lambda patterns: any(fnmatch.fnmatch(relative, p) or fnmatch.fnmatch(path.name, p) for p in patterns)
        if match(PATTERNS) and (checkpoints or not match(CHECKPOINTS)):
            found.append(path)
    return found


def running() -> bool:
    """True if a training command is running. A pgrep error counts as running, to be safe."""
    result = subprocess.run(["pgrep", "-f", RUNNING], capture_output=True)
    return getattr(result, "returncode", 0) != 1


def training_ended(runs: Path) -> bool:
    """Nothing is running and an end marker says the run is over."""
    return not running() and any((runs / marker).exists() for marker in ENDED)


def snapshot(runs: Path, files: list[Path]) -> Path:
    """Hard links to the files, so a checkpoint replaced during upload cannot break the commit."""
    folder = runs.parent / f".backup_snapshot_{runs.name}"
    shutil.rmtree(folder, ignore_errors=True)
    for path in files:
        target = folder / path.relative_to(runs)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    return folder


def upload(api, repo: str, runs: Path, machine: str, checkpoints: bool = True) -> int:
    files = wanted(runs, checkpoints)
    if not files:
        return 0
    folder = snapshot(runs, files)
    try:
        api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(folder), path_in_repo=machine,
                          allow_patterns=[glob.escape(p.relative_to(runs).as_posix()) for p in files],
                          commit_message=f"{machine}: backup of {len(files)} files")
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return len(files)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(api, repo: str, runs: Path, machine: str) -> list[str]:
    """Local files missing on the Hub or different there. Empty means all backed up.

    Files stored with LFS are compared by SHA-256, others by size.
    """
    remote = {}
    for entry in api.list_repo_tree(repo, path_in_repo=machine, recursive=True, repo_type="model"):
        if hasattr(entry, "size"):
            lfs = getattr(entry, "lfs", None)
            digest = None
            if lfs:
                digest = getattr(lfs, "sha256", None)
                if digest is None and isinstance(lfs, dict):
                    digest = lfs.get("sha256") or lfs.get("oid")
            remote[entry.path] = (entry.size, digest)
    problems = []
    for path in wanted(runs):
        key = f"{machine}/{path.relative_to(runs).as_posix()}"
        size, digest = remote.get(key, (None, None))
        if size != path.stat().st_size or (digest is not None and digest != sha256(path)):
            problems.append(key)
    return problems


def destroy_self(log) -> bool:
    """Ends this rental. The key goes to curl through standard input, so it never shows in `ps`."""
    key, instance = os.environ.get("CONTAINER_API_KEY"), os.environ.get("CONTAINER_ID")
    if not key or not instance:
        environ = dict(line.split("=", 1) for line in
                       Path("/proc/1/environ").read_bytes().decode(errors="ignore").split("\0") if "=" in line)
        key, instance = environ.get("CONTAINER_API_KEY"), environ.get("CONTAINER_ID")
    if not key or not instance:
        log("no per-instance key found; the laptop script must end this rental")
        return False
    result = subprocess.run(["curl", "-s", "--max-time", "30", "-w", "\n%{http_code}", "-X", "DELETE",
                             "-H", "@-", f"https://console.vast.ai/api/v0/instances/{instance}/"],
                            input=f"Authorization: Bearer {key}\n", capture_output=True, text=True, timeout=60)
    lines = (result.stdout or "").strip().splitlines() or [""]
    code, body = lines[-1].strip(), "\n".join(lines[:-1])
    log(f"destroy request answered {code}")
    return code.startswith("2") and '"success": false' not in body.replace("'", '"')


def wait_for_laptop(runs: Path, log, longest: int = 6 * 3600) -> None:
    """While the laptop is copying, it refreshes a marker file; wait for it, up to `longest`."""
    marker, waited = runs / LAPTOP_COPYING, 0
    while marker.exists() and time.time() - marker.stat().st_mtime < 900 and waited < longest:
        if waited == 0:
            log("the laptop is copying; waiting for it to finish")
        time.sleep(300)
        waited += 300


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--checkpoint-every", type=int, default=4, help="cycles between checkpoint uploads")
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

    seen_running, idle, cycle = False, 0, 0
    while True:
        busy = running()
        seen_running = seen_running or busy
        idle = 0 if busy else idle + 1
        ended = training_ended(runs) or (seen_running and idle >= 2)
        with_checkpoints = ended or cycle % args.checkpoint_every == 0
        try:
            log(f"uploaded {upload(api, args.repo, runs, args.machine, with_checkpoints)} files")
        except Exception as error:  # retried next cycle
            log(f"upload failed: {error}")
        if ended:
            break
        cycle += 1
        time.sleep(args.interval)

    # Final backup: quick retries first, then hourly for a day. Never destroy unverified.
    for attempt, pause in enumerate([300] * 6 + [3600] * 24, 1):
        try:
            upload(api, args.repo, runs, args.machine)
            problems = verify(api, args.repo, runs, args.machine)
        except Exception as error:
            problems = [f"error: {error}"]
        if not problems:
            log("final backup verified on the Hub")
            break
        log(f"verify attempt {attempt}: {len(problems)} files not backed up, e.g. {problems[:3]}")
        time.sleep(pause)
    else:
        log("backup could not be verified: not destroying this machine")
        return
    if args.no_destroy:
        return
    log(f"waiting {args.grace} s so the laptop can copy first")
    time.sleep(args.grace)
    wait_for_laptop(runs, log)
    for attempt in range(1, 97):  # every 30 minutes for up to 48 hours
        try:
            if destroy_self(log):
                return
        except Exception as error:
            log(f"destroy failed: {error}")
        time.sleep(1800)
    log("could not end this rental after 48 hours")


if __name__ == "__main__":
    main()
