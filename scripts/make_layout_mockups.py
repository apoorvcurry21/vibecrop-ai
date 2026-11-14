"""Draw the illustrated layout examples shown in the chat when the assistant asks "how should it look?".

Run once (the PNGs are committed):  .venv\\Scripts\\python.exe scripts\\make_layout_mockups.py
Output: src/assistant/static/layout_{overlay,split,pip,text}.png  (a 9:16 phone, 300x540)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "src" / "assistant" / "static"
W, H = 300, 540
SCALE = 3  # drawn 3x larger and shrunk, for smooth edges
BORDER = 6 * SCALE


def font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size * SCALE)
        except OSError:
            continue
    return ImageFont.load_default(size * SCALE)


def gradient(w: int, h: int, top: tuple, bottom: tuple) -> Image.Image:
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        d.line([(0, y), (w, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))
    return img


def broll_scene(w: int, h: int) -> Image.Image:
    """A generic 'footage' picture: sky, sun and two hills, so it reads as B-roll at a glance."""
    img = gradient(w, h, (92, 160, 235), (190, 225, 250))
    d = ImageDraw.Draw(img)
    r = min(w, h) // 7
    d.ellipse([w * 0.68 - r, h * 0.2 - r, w * 0.68 + r, h * 0.2 + r], fill=(255, 214, 90))
    d.polygon([(0, h), (0, h * 0.7), (w * 0.35, h * 0.5), (w * 0.7, h * 0.78), (w, h * 0.62), (w, h)], fill=(52, 140, 96))
    d.polygon([(0, h), (0, h * 0.86), (w * 0.45, h * 0.7), (w, h * 0.9), (w, h)], fill=(34, 105, 72))
    return img


def speaker_scene(w: int, h: int) -> Image.Image:
    """The talking head: a dark room with a head-and-shoulders silhouette."""
    img = gradient(w, h, (58, 62, 82), (30, 32, 44))
    d = ImageDraw.Draw(img)
    cx, head = w * 0.5, min(w, h) * 0.2
    top = h * 0.3
    d.ellipse([cx - head, top, cx + head, top + head * 2], fill=(214, 178, 150))
    d.rounded_rectangle([cx - head * 1.9, top + head * 2.15, cx + head * 1.9, h + head], radius=head,
                        fill=(70, 110, 190))
    return img


def phone(inner: Image.Image) -> Image.Image:
    """Put a scene inside a rounded phone outline on a transparent-ish light background."""
    canvas = Image.new("RGB", (W * SCALE + 2 * BORDER, H * SCALE + 2 * BORDER), (240, 242, 246))
    mask = Image.new("L", inner.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, inner.width - 1, inner.height - 1], radius=26 * SCALE, fill=255)
    canvas.paste(inner, (BORDER, BORDER), mask)
    ImageDraw.Draw(canvas).rounded_rectangle([BORDER // 2, BORDER // 2, canvas.width - BORDER // 2,
                                              canvas.height - BORDER // 2], radius=30 * SCALE,
                                             outline=(40, 44, 56), width=BORDER)
    return canvas.resize((canvas.width // SCALE, canvas.height // SCALE), Image.LANCZOS)


def caption_pill(img: Image.Image, y_frac: float = 0.8) -> None:
    d = ImageDraw.Draw(img)
    w, h = img.size
    f = font(17)
    text = "your captions"
    box = d.textbbox((0, 0), text, font=f)
    tw, th = box[2] - box[0], box[3] - box[1]
    x, y = (w - tw) / 2, h * y_frac
    d.rounded_rectangle([x - 14 * SCALE, y - 8 * SCALE, x + tw + 14 * SCALE, y + th + 12 * SCALE], radius=10 * SCALE,
                        fill=(0, 0, 0))
    d.text((x, y), text, font=f, fill=(255, 255, 255))


def inner() -> Image.Image:
    return Image.new("RGB", (W * SCALE, H * SCALE))


def overlay() -> Image.Image:
    img = broll_scene(W * SCALE, H * SCALE)
    caption_pill(img)
    return phone(img)


def split() -> Image.Image:
    img = inner()
    half = H * SCALE // 2
    img.paste(broll_scene(W * SCALE, half), (0, 0))
    img.paste(speaker_scene(W * SCALE, H * SCALE - half), (0, half))
    ImageDraw.Draw(img).rectangle([0, half - 4 * SCALE, W * SCALE, half + 4 * SCALE], fill=(255, 255, 255))
    caption_pill(img, 0.47)
    return phone(img)


def pip() -> Image.Image:
    img = speaker_scene(W * SCALE, H * SCALE)
    bw = int(W * SCALE * 0.58)
    bh = int(bw * 3 / 4)
    x, y = W * SCALE - bw - 16 * SCALE, 40 * SCALE
    img.paste(broll_scene(bw, bh), (x, y))
    ImageDraw.Draw(img).rectangle([x, y, x + bw, y + bh], outline=(255, 255, 255), width=3 * SCALE)
    caption_pill(img)
    return phone(img)


def text_example() -> Image.Image:
    img = speaker_scene(W * SCALE, H * SCALE)
    d = ImageDraw.Draw(img)
    f = font(46)
    text = "GOAT"
    box = d.textbbox((0, 0), text, font=f)
    x, y = (img.width - (box[2] - box[0])) / 2, 60 * SCALE
    d.text((x, y), text, font=f, fill=(255, 214, 10), stroke_width=4 * SCALE, stroke_fill=(0, 0, 0))
    caption_pill(img)
    return phone(img)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("overlay", overlay), ("split", split), ("pip", pip), ("text", text_example)):
        img = fn()
        img.save(OUT / f"layout_{name}.png", optimize=True)
        print("wrote", OUT / f"layout_{name}.png", img.size)


if __name__ == "__main__":
    main()
