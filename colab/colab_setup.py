"""Helpers for running VibeCrop on Google Colab (used by VibeCrop_Colab.ipynb; testable on any machine).

Colab gives a fresh disk every session, so the sound library and the noise-removal model live in a Google Drive folder,
while the temporary work files stay on the (fast) Colab disk."""
from __future__ import annotations

import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/apoorvcurry21/vibecrop-ai.git"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
PORT = 8501
SECRET_NAMES = ("GROQ_API_KEY", "GEMINI_API_KEY")
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def drive_env(drive_dir: str | Path) -> dict[str, str]:
    """Environment that keeps the big, slow-to-rebuild things on Drive and the scratch files on the Colab disk."""
    d = Path(drive_dir)
    return {"VIBECROP_SFX_DIR": str(d / "sfx"), "VIBECROP_MODELS_DIR": str(d / "models"),
            "VIBECROP_WORKDIR": "/content/vibecrop_work"}


def load_secrets(get=None) -> tuple[dict[str, str], list[str]]:
    """The API keys from Colab Secrets (`get` = google.colab.userdata.get). Returns (found, missing names).
    They must go into environment variables: Streamlit runs as a separate process, which cannot read Colab Secrets."""
    if get is None:
        from google.colab import userdata
        get = userdata.get
    found, missing = {}, []
    for name in SECRET_NAMES:
        try:
            value = (get(name) or "").strip()
        except Exception:  # noqa: BLE001 - SecretNotFoundError / NotebookAccessError
            value = ""
        if value:
            found[name] = value
        else:
            missing.append(name)
    return found, missing


def parse_tunnel_url(line: str) -> str | None:
    m = TUNNEL_RE.search(line or "")
    return m.group(0) if m else None


def streamlit_command(repo: str | Path, port: int = PORT) -> list[str]:
    """Behind a tunnel the page's origin differs from the server's, so the origin and XSRF checks must be off or file
    uploads are rejected (403)."""
    return ["streamlit", "run", str(Path(repo) / "app.py"), "--server.port", str(port), "--server.headless", "true",
            "--server.enableCORS", "false", "--server.enableXsrfProtection", "false",
            "--browser.gatherUsageStats", "false"]


def ensure_cloudflared(dest: str | Path = "/content/cloudflared") -> Path:
    dest = Path(dest)
    if not dest.is_file():
        urllib.request.urlretrieve(CLOUDFLARED_URL, dest)
        dest.chmod(0o755)
    return dest


def start_tunnel(port: int = PORT, timeout: float = 60.0, cloudflared: str | Path | None = None):
    """Start a quick Cloudflare tunnel (no account needed). Returns (process, public url)."""
    binary = str(cloudflared or ensure_cloudflared())
    proc = subprocess.Popen([binary, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + timeout
    for line in proc.stdout:  # type: ignore[union-attr]
        url = parse_tunnel_url(line)
        if url:
            return proc, url
        if time.time() > deadline:
            break
    proc.kill()
    raise RuntimeError("The tunnel did not start. Run this cell again.")


def launch(repo: str | Path, env: dict[str, str]):
    """Start Streamlit and the tunnel. Returns (streamlit process, tunnel process, public url)."""
    full_env = {**os.environ, **env}
    app = subprocess.Popen(streamlit_command(repo), env=full_env, cwd=str(repo), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    time.sleep(4)
    tunnel, url = start_tunnel()
    return app, tunnel, url
