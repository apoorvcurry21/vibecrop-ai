"""Crash-safe saves: plan.json / job.json must never be left half-written."""
import json
import os
import threading

import pytest

from src import fileio, workflow


def test_write_replaces_the_file_and_leaves_no_temp_files(tmp_path):
    f = tmp_path / "plan.json"
    fileio.write_json_atomic(f, {"a": 1})
    fileio.write_json_atomic(f, {"a": 2, "name": "हिन्दी"})
    assert json.loads(f.read_text(encoding="utf-8")) == {"a": 2, "name": "हिन्दी"}
    assert [p.name for p in tmp_path.iterdir()] == ["plan.json"]


def test_a_failed_save_keeps_the_old_file_complete(tmp_path, monkeypatch):
    f = tmp_path / "plan.json"
    fileio.write_json_atomic(f, {"version": "old"})

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        fileio.write_json_atomic(f, {"version": "new"})
    monkeypatch.undo()
    assert json.loads(f.read_text(encoding="utf-8")) == {"version": "old"}
    assert [p.name for p in tmp_path.iterdir()] == ["plan.json"]  # the temp file is cleaned up too


def test_a_brief_windows_lock_is_retried(tmp_path, monkeypatch):
    f = tmp_path / "job.json"
    real, calls = os.replace, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("file is in use")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    fileio.write_json_atomic(f, {"ok": True})
    assert calls["n"] == 3 and json.loads(f.read_text(encoding="utf-8")) == {"ok": True}


def test_a_reader_never_sees_a_torn_file_while_another_thread_saves(tmp_path):
    f = tmp_path / "plan.json"
    fileio.write_json_atomic(f, {"items": list(range(2000))})
    stop, bad = threading.Event(), []

    def writer():
        n = 0
        while not stop.is_set():
            n += 1
            fileio.write_json_atomic(f, {"items": list(range(2000)), "n": n})

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(200):
            try:
                json.loads(f.read_text(encoding="utf-8"))
            except PermissionError:
                continue  # Windows: the file was being swapped at that instant (not a torn read)
            except ValueError as exc:
                bad.append(exc)
    finally:
        stop.set()
        t.join()
    assert not bad


def test_workflow_saves_go_through_the_atomic_writer(tmp_path, monkeypatch):
    job = tmp_path / "jobs" / "j"
    job.mkdir(parents=True)
    seen = []
    real = fileio.write_text_atomic
    monkeypatch.setattr("src.fileio.write_text_atomic", lambda p, t, *a, **k: (seen.append(os.path.basename(p)), real(p, t, *a, **k)))
    workflow.write_job_meta(job, language="en")
    workflow._save(job, {"items": []})
    assert seen == ["job.json", "plan.json"]
