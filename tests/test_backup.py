import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("backup_to_hub", Path(__file__).parents[1] / "scripts" / "backup_to_hub.py")
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


class FakeHub:
    def __init__(self):
        self.files = {}
        self.uploads = 0

    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, allow_patterns, commit_message):
        self.uploads += 1
        for relative in allow_patterns:
            self.files[f"{path_in_repo}/{relative}"] = (Path(folder_path) / relative).stat().st_size

    def list_repo_tree(self, repo, path_in_repo, recursive, repo_type):
        return [SimpleNamespace(path=p, size=s) for p, s in self.files.items() if p.startswith(path_in_repo + "/")]


@pytest.fixture
def runs(tmp_path):
    (tmp_path / "plain_warm_s0").mkdir()
    (tmp_path / "plain_warm_s0" / "last.pt").write_bytes(b"x" * 100)
    (tmp_path / "plain_warm_s0" / "step_20000.pt").write_bytes(b"x" * 100)
    (tmp_path / "plain_warm_s0" / "last.tmp").write_bytes(b"x")
    (tmp_path / "plain_warm_s0" / "log.jsonl").write_text("{}\n")
    (tmp_path / "results_v3.json").write_text("{}")
    (tmp_path / "run_v3.log").write_text("ok")
    return tmp_path


def test_backs_up_logs_results_and_latest_checkpoints_only(runs):
    names = {p.relative_to(runs).as_posix() for p in backup.wanted(runs)}
    assert names == {"plain_warm_s0/last.pt", "plain_warm_s0/log.jsonl", "results_v3.json", "run_v3.log"}


def test_verify_finds_missing_and_changed_files(runs):
    hub = FakeHub()
    assert backup.upload(hub, "u/spidory-runs", runs, "seed0") == 4
    assert backup.verify(hub, "u/spidory-runs", runs, "seed0") == []
    (runs / "run_v3.log").write_text("longer now")
    (runs / "DONE").write_text("")
    problems = backup.verify(hub, "u/spidory-runs", runs, "seed0")
    assert set(problems) == {"seed0/run_v3.log", "seed0/DONE"}


def test_training_ended_by_marker_or_no_process(runs, monkeypatch):
    monkeypatch.setattr(backup.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert not backup.training_ended(runs)  # a training process is running
    (runs / "STOPPED").write_text("x")
    assert not backup.training_ended(runs)  # a marker while training runs is stale: training wins
    monkeypatch.setattr(backup.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    assert backup.training_ended(runs)  # nothing running and a marker: ended
    (runs / "STOPPED").unlink()
    monkeypatch.setattr(backup.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    assert not backup.training_ended(runs)  # no marker: a crash is handled by the 2-check rule in main


def test_never_destroys_without_a_verified_backup(runs, monkeypatch):
    calls = []
    monkeypatch.setattr(backup, "destroy_self", lambda log: calls.append(1))
    monkeypatch.setattr(backup, "training_ended", lambda r: True)
    monkeypatch.setattr(backup, "verify", lambda *a: ["seed0/x"])
    monkeypatch.setattr(backup.time, "sleep", lambda s: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeHub))
    monkeypatch.setattr(sys, "argv", ["b", "--repo", "u/r", "--machine", "seed0", "--runs", str(runs)])
    backup.main()
    assert calls == []
    assert "not destroying" in (runs / "backup_to_hub.log").read_text()


def test_destroys_after_verified_backup_and_grace(runs, monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(backup, "destroy_self", lambda log: calls.append(1) or True)
    monkeypatch.setattr(backup, "training_ended", lambda r: True)
    monkeypatch.setattr(backup.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeHub))
    monkeypatch.setattr(sys, "argv", ["b", "--repo", "u/r", "--machine", "seed0", "--runs", str(runs), "--grace", "3600"])
    backup.main()
    assert calls == [1] and 3600 in sleeps


def test_destroy_uses_only_this_instance(monkeypatch):
    seen = {}
    monkeypatch.setenv("CONTAINER_API_KEY", "k")
    monkeypatch.setenv("CONTAINER_ID", "123")
    monkeypatch.setattr(backup.subprocess, "run",
                        lambda cmd, **k: (seen.update(cmd=cmd), SimpleNamespace(stdout="200"))[1])
    assert backup.destroy_self(lambda t: None)
    assert seen["cmd"][-1] == "https://console.vast.ai/api/v0/instances/123/"
    assert "DELETE" in seen["cmd"]


def test_a_crash_ends_only_after_two_idle_checks_once_training_was_seen(runs, monkeypatch):
    destroyed = []
    monkeypatch.setattr(backup, "destroy_self", lambda log: destroyed.append(1) or True)
    codes = iter([0, 0, 1, 1])  # running, running, then nothing twice: crashed
    monkeypatch.setattr(backup.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=next(codes, 1)))
    monkeypatch.setattr(backup.time, "sleep", lambda s: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeHub))
    monkeypatch.setattr(sys, "argv", ["b", "--repo", "u/r", "--machine", "seed0", "--runs", str(runs)])
    backup.main()
    assert destroyed == [1]


def test_waits_while_the_laptop_is_copying(runs, monkeypatch):
    (runs / backup.LAPTOP_COPYING).write_text("")
    sleeps = []
    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:
            (runs / backup.LAPTOP_COPYING).unlink()  # the laptop finished
    monkeypatch.setattr(backup.time, "sleep", sleep)
    backup.wait_for_laptop(runs, lambda t: None)
    assert sleeps == [300, 300, 300]


def test_checkpoints_upload_less_often_than_logs(runs):
    names = {p.name for p in backup.wanted(runs, checkpoints=False)}
    assert "last.pt" not in names and "log.jsonl" in names
