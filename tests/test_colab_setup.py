"""The Colab helpers and the notebook (nothing here needs Colab)."""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "colab"))
import colab_setup  # noqa: E402

NOTEBOOK = ROOT / "colab" / "VibeCrop_Colab.ipynb"


def test_drive_holds_the_slow_to_rebuild_things_and_the_colab_disk_holds_the_scratch_files():
    env = colab_setup.drive_env("/content/drive/MyDrive/VibeCrop")
    assert env["VIBECROP_SFX_DIR"].replace("\\", "/") == "/content/drive/MyDrive/VibeCrop/sfx"
    assert env["VIBECROP_MODELS_DIR"].replace("\\", "/") == "/content/drive/MyDrive/VibeCrop/models"
    assert env["VIBECROP_WORKDIR"] == "/content/vibecrop_work"  # temp media on the fast local disk, not on Drive


def test_secrets_are_returned_for_the_environment_and_missing_ones_are_named():
    values = {"GROQ_API_KEY": "  gsk_abc  ", "GEMINI_API_KEY": ""}
    found, missing = colab_setup.load_secrets(lambda name: values[name])
    assert found == {"GROQ_API_KEY": "gsk_abc"} and missing == ["GEMINI_API_KEY"]

    def boom(name):
        raise RuntimeError("SecretNotFoundError")

    assert colab_setup.load_secrets(boom) == ({}, ["GROQ_API_KEY", "GEMINI_API_KEY"])


def test_the_tunnel_url_is_found_in_cloudflareds_output():
    line = "2026-11-15T10:00:00Z INF |  https://quiet-river-fox-1234.trycloudflare.com  |"
    assert colab_setup.parse_tunnel_url(line) == "https://quiet-river-fox-1234.trycloudflare.com"
    assert colab_setup.parse_tunnel_url("INF Requesting new quick Tunnel on trycloudflare.com...") is None
    assert colab_setup.parse_tunnel_url("") is None and colab_setup.parse_tunnel_url(None) is None


def test_streamlit_is_started_so_uploads_work_behind_a_tunnel():
    cmd = colab_setup.streamlit_command("/content/vibecrop-ai")
    assert cmd[:2] == ["streamlit", "run"] and cmd[2].replace("\\", "/") == "/content/vibecrop-ai/app.py"
    flags = dict(zip(cmd[3::2], cmd[4::2]))
    assert flags["--server.enableXsrfProtection"] == "false" and flags["--server.enableCORS"] == "false"
    assert flags["--server.headless"] == "true" and flags["--server.port"] == "8501"


def test_start_tunnel_returns_the_first_url_and_gives_up_cleanly(monkeypatch):
    started = []

    class FakeProc:
        def __init__(self, cmd, **kw):
            started.append(cmd)
            self.stdout = iter(["INF Starting tunnel\n", "INF |  https://abc-def.trycloudflare.com  |\n", "more\n"])
            self.killed = False

        def kill(self):
            self.killed = True

    monkeypatch.setattr(colab_setup.subprocess, "Popen", FakeProc)
    proc, url = colab_setup.start_tunnel(cloudflared="/fake/cloudflared")
    assert url == "https://abc-def.trycloudflare.com" and started[0][:2] == ["/fake/cloudflared", "tunnel"]
    assert "http://localhost:8501" in started[0]

    class Silent(FakeProc):
        def __init__(self, cmd, **kw):
            super().__init__(cmd, **kw)
            self.stdout = iter(["nothing useful\n"] * 3)

    monkeypatch.setattr(colab_setup.subprocess, "Popen", Silent)
    with pytest.raises(RuntimeError, match="tunnel did not start"):
        colab_setup.start_tunnel(cloudflared="/fake/cloudflared", timeout=-1)


def _code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return nb, [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


def test_the_notebook_is_valid_ordered_and_contains_no_secrets():
    nb, code = _code_cells()
    assert nb["nbformat"] == 4 and len(code) == 5
    for needle, cell in zip(("drive.mount", "apt-get", "load_secrets", "build_sfx_library", "colab_setup.launch"), code):
        assert needle in cell
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert not re.search(r"gsk_[A-Za-z0-9]{10,}|AIza[A-Za-z0-9_-]{20,}", text)  # no real key ever lands in the repo
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in nb["cells"] if c["cell_type"] == "code")


def test_every_python_line_of_the_notebook_compiles():
    _, code = _code_cells()
    for i, cell in enumerate(code, 1):
        py = "\n".join(line[:len(line) - len(line.lstrip())] + "pass" if line.lstrip().startswith("!") else line
                       for line in cell.splitlines())  # keep the indentation of the replaced shell lines
        compile(py, f"cell{i}", "exec")  # shell lines (!) are IPython; the rest must be plain, valid Python


def test_the_repo_url_and_requirements_match_the_notebook():
    _, code = _code_cells()
    assert colab_setup.REPO_URL.endswith("apoorvcurry21/vibecrop-ai.git") and "vibecrop-ai.git" in code[1]
    assert "requirements.txt" in code[1] and "scripts/build_sfx_library.py" in code[3]
    assert (ROOT / "requirements.txt").is_file() and (ROOT / "scripts" / "build_sfx_library.py").is_file()
