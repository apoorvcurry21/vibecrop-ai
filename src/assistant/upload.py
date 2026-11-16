"""The 'Add picture / video' panel on the review page: put a file of the user's own on the timeline with no chat and no
AI call. It runs the same tools the chat uses (place_image / place_video_file), so the rules, the undo snapshot and the
rollback on failure are identical."""
from __future__ import annotations

from pathlib import Path

from src.assistant import assets, tools
from src.assistant.tools import ToolContext

FILE_TYPES = ["png", "jpg", "jpeg", "webp", "mp4", "mov", "m4v"]
DEFAULT_LENGTH_S = 3.0


class UploadError(RuntimeError):
    pass


def add_upload(job: Path, filename: str, data: bytes, start: float, length: float = DEFAULT_LENGTH_S,
               layout: str = "overlay", position: str | None = None, size: float | None = None,
               from_second: float = 0.0) -> tuple[dict, list[str]]:
    """Save the uploaded file and place it as a new slot [start, start + length]. Returns (the slot, notes such as
    'this covers slot 3'). Raises UploadError with a plain message; on failure the project is left as it was."""
    try:
        att = assets.save_attachment(job, filename, data)
    except assets.AssetError as exc:
        raise UploadError(str(exc)) from None
    args: dict = {"start": float(start), "end": float(start) + float(length), "layout": layout}
    if layout == "pip":
        args.update({"position": position, "size": size})
    if att.kind == "image":
        name, args["image"] = "place_image", att.id
    else:
        name, args["video"], args["from_second"] = "place_video_file", att.id, float(from_second)
    result = tools.run_tool(ToolContext(job=job, attachments={att.id: att}), name, args)
    if "error" in result:
        raise UploadError(result["error"])
    return result["slot"], list(result.get("notes") or [])
