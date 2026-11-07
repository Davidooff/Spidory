"""Review tests for scripts/backup_to_hub.py: the risky cases of backing up and ending a rental.

No network, no real Hub, no Vast calls. Tests marked xfail(strict=True) show a real problem in the
current script; they start passing (and then fail as XPASS) once it is fixed.
"""
import hashlib
import importlib.util
import re
import subprocess
import sys
import time as real_time
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.hf_api import RepoFile, RepoFolder

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("backup_to_hub_review", ROOT / "scripts" / "backup_to_hub.py")
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)

MACHINES = ("v3_seed0", "v3_seed1", "between")


def backup_cmdline(machine):
    return f"/venv/main/bin/python scripts/backup_to_hub.py --repo Davidow/spidory-runs --machine {machine} --runs runs"


class FakeHub:
    """Keeps size and sha256 per path, like the Hub's tree listing (RepoFile.size, RepoFile.lfs.sha256)."""

    def __init__(self, fail_uploads=0):
        self.files = {}
        self.uploads = 0
        self.fail_uploads = fail_uploads
        self.messages = []

    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, allow_patterns, commit_message):
        self.uploads += 1
        self.messages.append(commit_message)
        if self.uploads <= self.fail_uploads:
            raise RuntimeError("429 Too Many Requests")
        for relative in allow_patterns:
            data = (Path(folder_path) / relative).read_bytes()
            self.files[f"{path_in_repo}/{relative}"] = (len(data), hashlib.sha256(data).hexdigest())

    def list_repo_tree(self, repo, path_in_repo, recursive, repo_type):
        entries = [RepoFolder(path=path_in_repo, oid="t0")]
        for path, (size, sha) in self.files.items():
            if path.startswith(path_in_repo + "/"):
                entries.append(RepoFile(path=path, size=size, oid="b",
                                        lfs={"size": size, "oid": sha, "pointerSize": 134}))
        return entries


@pytest.fixture
def runs(tmp_path):
    (tmp_path / "plain_warm_s0").mkdir()
    (tmp_path / "plain_warm_s0" / "last.pt").write_bytes(b"a" * 100)
    (tmp_path / "plain_warm_s0" / "log.jsonl").write_text("{}\n")
    (tmp_path / "run_v3.log").write_text("ok")
    return tmp_path


def run_main(monkeypatch, runs, hub, *extra):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: hub))
    monkeypatch.setattr(sys, "argv", ["b", "--repo", "u/r", "--machine", "seed0", "--runs", str(runs), *extra])
    backup.main()


class Stop(Exception):
    pass


# ---------------------------------------------------------------- the "is training running?" pattern

@pytest.mark.parametrize("cmdline", [
    "bash scripts/run_v3.sh 0",
    "/bin/bash scripts/run_v3.sh 1",
    "bash scripts/run_between.sh 0",
    "/bin/bash scripts/build_then_between.sh 0",
    "/venv/main/bin/python -m spidory.train --name plain_warm_s0 --plain --stage1 24000",
    "/venv/main/bin/python -m spidory.rounds --name plain_s0 --warm plain_warm_s0 --control-only",
    "/venv/main/bin/python -m spidory.evaluate --seeds 0 --names plain constant",
    "/venv/main/bin/python -m spidory.clean_v5 --root data",
    # A chained launch: the parent shell holds the whole line, so the gap between commands is covered.
    "bash -c /venv/main/bin/python -m spidory.clean_v5 --root data && scripts/run_v3.sh 0",
])
def test_pattern_sees_every_training_command(cmdline):
    assert re.search(backup.RUNNING, cmdline)


@pytest.mark.parametrize("cmdline", [
    *(backup_cmdline(m) for m in MACHINES),
    "pgrep -f " + backup.RUNNING,  # the check itself
    # The laptop's remote check (bash -c over ssh) and the watchdog must not count as training.
    'bash -c cd /workspace/Spidory/runs\n if pgrep -f "[r]un_v3.sh|[b]uild_v4_then_check.sh|'
    '[s]pidory\\.(train|rounds|evaluate|clean_v5)" >/dev/null; then echo RUNNING',
    "bash scripts/watchdog.sh 72000",
    "sleep 72000",
    "tail -f runs/run_v3.log",
    "tail -f runs/run_between.log",
])
def test_pattern_ignores_backup_script_checks_and_watchers(cmdline):
    assert not re.search(backup.RUNNING, cmdline)


def test_watchdog_kill_patterns_spare_the_backup_script():
    text = (ROOT / "scripts" / "watchdog.sh").read_text()
    patterns = re.findall(r"pkill -f '([^']+)'", text)
    assert patterns
    for machine in MACHINES:
        assert not any(re.search(p, backup_cmdline(machine)) for p in patterns)


def test_real_pgrep_sees_a_process_shaped_like_the_run_script(tmp_path):
    # A harmless child whose command line contains "scripts/run_v3.sh"; it ends on its own.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)", "scripts/run_v3.sh", "0"])
    try:
        real_time.sleep(0.2)
        assert backup.training_ended(tmp_path) is False
    finally:
        child.wait(timeout=10)


def test_real_pgrep_does_not_see_the_backup_script(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)",
                              *backup_cmdline("v3_seed0").split()[1:]])
    try:
        real_time.sleep(0.2)
        found = subprocess.run(["pgrep", "-f", backup.RUNNING], capture_output=True, text=True).stdout.split()
        assert str(child.pid) not in found
    finally:
        child.wait(timeout=10)


def test_data_growth_on_its_own_counts_as_running():
    assert re.search(backup.RUNNING, "/venv/main/bin/python -m spidory.grow_v4 --root data")


# ---------------------------------------------------------------- ending too early

def test_stale_marker_does_not_end_a_live_run(runs, monkeypatch):
    (runs / "STOPPED").write_text("early check failed")  # left over from the stopped first start
    monkeypatch.setattr(backup.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert backup.training_ended(runs) is False


def test_pgrep_error_is_not_read_as_nothing_running(runs, monkeypatch):
    monkeypatch.setattr(backup.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=3))
    assert backup.training_ended(runs) is False


def _pgrep_sequence(monkeypatch, codes):
    """Fake pgrep: returns the given exit codes in order, then the last one forever."""
    calls = []

    def run(cmd, **kwargs):
        assert cmd[0] == "pgrep", f"unexpected subprocess call {cmd}"
        calls.append(1)
        return SimpleNamespace(returncode=codes[min(len(calls), len(codes)) - 1])
    monkeypatch.setattr(backup.subprocess, "run", run)


def _stop_after_training_sleeps(monkeypatch, interval=1800, limit=3):
    count = []

    def sleep(seconds):
        if seconds == interval:
            count.append(1)
            if len(count) >= limit:
                raise Stop
    monkeypatch.setattr(backup.time, "sleep", sleep)


def test_one_momentary_gap_does_not_destroy(runs, monkeypatch):
    destroyed = []
    monkeypatch.setattr(backup, "destroy_self", lambda log: destroyed.append(1) or True)
    _pgrep_sequence(monkeypatch, [1, 0])  # a gap at this check, training running afterwards
    _stop_after_training_sleeps(monkeypatch)
    with pytest.raises(Stop):
        run_main(monkeypatch, runs, FakeHub())
    assert destroyed == []


def test_started_before_training_does_not_destroy(runs, monkeypatch):
    destroyed = []
    monkeypatch.setattr(backup, "destroy_self", lambda log: destroyed.append(1) or True)
    _pgrep_sequence(monkeypatch, [1, 1, 0])  # training starts some minutes later
    _stop_after_training_sleeps(monkeypatch)
    with pytest.raises(Stop):
        run_main(monkeypatch, runs, FakeHub())
    assert destroyed == []


def test_keeps_backing_up_while_training_runs(runs, monkeypatch):
    destroyed = []
    monkeypatch.setattr(backup, "destroy_self", lambda log: destroyed.append(1) or True)
    _pgrep_sequence(monkeypatch, [0])
    _stop_after_training_sleeps(monkeypatch)
    hub = FakeHub()
    with pytest.raises(Stop):
        run_main(monkeypatch, runs, hub)
    assert destroyed == [] and hub.uploads == 3


def test_done_marker_written_during_the_wait_is_uploaded_before_destroy(runs, monkeypatch):
    destroyed = []
    hub = FakeHub()
    monkeypatch.setattr(backup, "destroy_self", lambda log: destroyed.append(dict(hub.files)) or True)
    state = {"done": False}
    monkeypatch.setattr(backup, "training_ended", lambda r: state["done"])

    def sleep(seconds):
        if seconds == 1800:
            (runs / "plain_warm_s0" / "last.pt").write_bytes(b"b" * 100)  # final checkpoint, same size
            (runs / "DONE").write_text("")
            state["done"] = True
    monkeypatch.setattr(backup.time, "sleep", sleep)
    run_main(monkeypatch, runs, hub)
    on_hub = destroyed[0]
    assert "seed0/DONE" in on_hub
    assert on_hub["seed0/plain_warm_s0/last.pt"][1] == hashlib.sha256(b"b" * 100).hexdigest()


# ---------------------------------------------------------------- verify

def test_verify_skips_folders_and_uses_full_lfs_size(runs):
    hub = FakeHub()
    backup.upload(hub, "u/r", runs, "seed0")
    entries = hub.list_repo_tree("u/r", path_in_repo="seed0", recursive=True, repo_type="model")
    assert any(not hasattr(e, "size") for e in entries)  # a RepoFolder is listed too
    assert backup.verify(hub, "u/r", runs, "seed0") == []


def test_verify_does_not_mix_machines(runs):
    hub = FakeHub()
    backup.upload(hub, "u/r", runs, "seed1")  # another machine's folder
    assert set(backup.verify(hub, "u/r", runs, "seed0")) == {
        "seed0/plain_warm_s0/last.pt", "seed0/plain_warm_s0/log.jsonl", "seed0/run_v3.log"}


def test_verify_catches_an_older_checkpoint_of_the_same_size(runs):
    hub = FakeHub()
    backup.upload(hub, "u/r", runs, "seed0")
    (runs / "plain_warm_s0" / "last.pt").write_bytes(b"b" * 100)  # newer weights, same size
    assert backup.verify(hub, "u/r", runs, "seed0") == ["seed0/plain_warm_s0/last.pt"]


def test_file_growing_after_upload_is_retried_before_destroy(runs, monkeypatch):
    destroyed, sleeps = [], []
    monkeypatch.setattr(backup, "destroy_self", lambda log: destroyed.append(1) or True)
    monkeypatch.setattr(backup, "training_ended", lambda r: True)
    monkeypatch.setattr(backup.time, "sleep", sleeps.append)
    real_verify, calls = backup.verify, []

    def verify(*args):
        calls.append(1)
        if len(calls) == 1:  # a log line arrives between upload and verify
            with open(runs / "run_v3.log", "a") as sink:
                sink.write(" late line")
        return real_verify(*args)
    monkeypatch.setattr(backup, "verify", verify)
    run_main(monkeypatch, runs, FakeHub())
    assert len(calls) == 2 and 300 in sleeps and destroyed == [1]


def test_file_names_with_glob_characters_are_uploaded(tmp_path):
    from huggingface_hub.utils import filter_repo_objects
    (tmp_path / "run[1]").mkdir()
    (tmp_path / "run[1]" / "log.jsonl").write_text("{}\n")

    class FilteringHub(FakeHub):
        def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, allow_patterns, commit_message):
            every = [p.relative_to(folder_path).as_posix() for p in Path(folder_path).rglob("*") if p.is_file()]
            super().upload_folder(repo_id, repo_type, folder_path, path_in_repo,
                                  list(filter_repo_objects(every, allow_patterns=allow_patterns)), commit_message)
    hub = FilteringHub()
    backup.upload(hub, "u/r", tmp_path, "seed0")
    assert backup.verify(hub, "u/r", tmp_path, "seed0") == []


# ---------------------------------------------------------------- Hub errors

def test_hub_errors_during_training_are_logged_and_retried(runs, monkeypatch):
    destroyed = []
    monkeypatch.setattr(backup, "destroy_self", lambda log: destroyed.append(1) or True)
    ends = iter([False, False, True])
    monkeypatch.setattr(backup, "training_ended", lambda r: next(ends))
    monkeypatch.setattr(backup.time, "sleep", lambda s: None)
    hub = FakeHub(fail_uploads=2)
    run_main(monkeypatch, runs, hub)
    text = (runs / "backup_to_hub.log").read_text()
    assert text.count("upload failed: 429") == 2 and destroyed == [1]


def test_a_long_hub_outage_at_the_end_keeps_the_machine(runs, monkeypatch):
    # Never destroys unverified: quick retries, then hourly for a day, then it leaves the rental to the laptop.
    destroyed, sleeps = [], []
    monkeypatch.setattr(backup, "destroy_self", lambda log: destroyed.append(1) or True)
    monkeypatch.setattr(backup, "training_ended", lambda r: True)
    monkeypatch.setattr(backup.time, "sleep", sleeps.append)
    hub = FakeHub(fail_uploads=10 ** 6)
    run_main(monkeypatch, runs, hub)
    assert destroyed == [] and sleeps == [300] * 6 + [3600] * 24  # retries for a day, never destroys unverified
    assert "not destroying" in (runs / "backup_to_hub.log").read_text()


def test_no_destroy_flag_never_destroys(runs, monkeypatch):
    destroyed = []
    monkeypatch.setattr(backup, "destroy_self", lambda log: destroyed.append(1) or True)
    monkeypatch.setattr(backup, "training_ended", lambda r: True)
    monkeypatch.setattr(backup.time, "sleep", lambda s: None)
    run_main(monkeypatch, runs, FakeHub(), "--no-destroy")
    assert destroyed == []


# ---------------------------------------------------------------- destroy

def test_destroy_reads_proc1_environ_when_the_shell_lacks_the_key(monkeypatch):
    seen = {}
    monkeypatch.delenv("CONTAINER_API_KEY", raising=False)
    monkeypatch.delenv("CONTAINER_ID", raising=False)
    monkeypatch.setattr(backup, "Path", lambda p: SimpleNamespace(
        read_bytes=lambda: b"PATH=/bin\0CONTAINER_API_KEY=k=with=equals\0CONTAINER_ID=42\0"))
    monkeypatch.setattr(backup.subprocess, "run",
                        lambda cmd, **k: (seen.update(cmd=cmd, kwargs=k), SimpleNamespace(stdout="200"))[1])
    assert backup.destroy_self(lambda t: None)
    assert seen["cmd"][-1].endswith("/instances/42/")
    assert "Authorization: Bearer k=with=equals" in seen["kwargs"]["input"]  # through stdin, not argv
    assert not any("k=with=equals" in part for part in seen["cmd"])


def test_destroy_without_a_key_does_nothing(monkeypatch):
    monkeypatch.delenv("CONTAINER_API_KEY", raising=False)
    monkeypatch.delenv("CONTAINER_ID", raising=False)
    monkeypatch.setattr(backup, "Path", lambda p: SimpleNamespace(read_bytes=lambda: b"PATH=/bin\0"))
    monkeypatch.setattr(backup.subprocess, "run", lambda *a, **k: pytest.fail("no request without a key"))
    logs = []
    assert backup.destroy_self(logs.append) is False and "laptop" in logs[0]


def test_secrets_never_reach_logs_or_commit_messages(runs, monkeypatch):
    monkeypatch.setenv("CONTAINER_API_KEY", "SECRET-VAST-KEY")
    monkeypatch.setenv("CONTAINER_ID", "7")
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETTOKEN")
    monkeypatch.setattr(backup, "training_ended", lambda r: True)
    monkeypatch.setattr(backup.time, "sleep", lambda s: None)
    monkeypatch.setattr(backup.subprocess, "run", lambda cmd, **k: SimpleNamespace(stdout="401"))
    hub = FakeHub()
    run_main(monkeypatch, runs, hub)
    text = (runs / "backup_to_hub.log").read_text() + "".join(hub.messages)
    assert "SECRET" not in text and "destroy request answered 401" in text


def test_failed_destroy_is_retried(runs, monkeypatch):
    attempts = []
    monkeypatch.setattr(backup, "destroy_self", lambda log: attempts.append(1) or len(attempts) >= 2)
    monkeypatch.setattr(backup, "training_ended", lambda r: True)
    monkeypatch.setattr(backup.time, "sleep", lambda s: None)
    run_main(monkeypatch, runs, FakeHub())
    assert len(attempts) == 2


def test_destroy_request_has_a_time_limit(monkeypatch):
    seen = {}
    monkeypatch.setenv("CONTAINER_API_KEY", "k")
    monkeypatch.setenv("CONTAINER_ID", "1")
    monkeypatch.setattr(backup.subprocess, "run",
                        lambda cmd, **k: (seen.update(cmd=cmd, kwargs=k), SimpleNamespace(stdout="200"))[1])
    backup.destroy_self(lambda t: None)
    assert "--max-time" in seen["cmd"] or "-m" in seen["cmd"] or "timeout" in seen["kwargs"]
