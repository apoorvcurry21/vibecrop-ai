"""Download an image from a URL the user (or a web search) supplied, safely: http(s) only, never a private or local
address (the app runs on the user's own PC, so `http://192.168.x.x/` or `localhost` must stay unreachable), every
redirect re-checked, a hard size cap and a short timeout."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

MAX_IMAGE_BYTES = 15 * 1024 * 1024
TIMEOUT_S = 12
MAX_REDIRECTS = 4
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; VibeCropAI/1.0)", "Accept": "image/*,*/*;q=0.5"}


class FetchError(RuntimeError):
    pass


def check_url(url: str) -> str:
    """Raise FetchError unless `url` is an http(s) URL whose host resolves only to public addresses."""
    parts = urlparse((url or "").strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("Only normal http(s) links are allowed.")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise FetchError(f"Could not find the site '{parts.hostname}'.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
                or ip.is_unspecified):
            raise FetchError("That link points to a private or local address, so I won't open it.")
    return parts.geturl()


def fetch_image(url: str, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """The raw bytes of an image at `url` (the caller still validates them as a real image)."""
    import requests

    current = url
    for _ in range(MAX_REDIRECTS + 1):
        check_url(current)
        try:
            resp = requests.get(current, headers=HEADERS, timeout=TIMEOUT_S, stream=True, allow_redirects=False)
        except requests.RequestException as exc:
            raise FetchError(f"Could not download the image: {type(exc).__name__}.") from exc
        with resp:
            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                nxt = resp.headers.get("Location")
                if not nxt:
                    raise FetchError("The link redirects nowhere.")
                current = urljoin(current, nxt)
                continue
            if resp.status_code != 200:
                raise FetchError(f"The site answered {resp.status_code}.")
            kind = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if not kind.startswith("image/"):
                raise FetchError("That link is not a picture (it is a web page or another file type).")
            size = int(resp.headers.get("Content-Length") or 0)
            if size > max_bytes:
                raise FetchError("That image is too big (over 15 MB).")
            data = bytearray()
            for block in resp.iter_content(64 * 1024):
                data += block
                if len(data) > max_bytes:
                    raise FetchError("That image is too big (over 15 MB).")
            if not data:
                raise FetchError("The image was empty.")
            return bytes(data)
    raise FetchError("Too many redirects.")
