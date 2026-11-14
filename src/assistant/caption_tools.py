"""Chat tools for captions: pick a caption style, and make chosen words stand out (another colour and/or bigger).
Registered in tools.REGISTRY when this module is imported (tools.py does that at its end)."""
from __future__ import annotations

from src import workflow
from src.assistant import phrases
from src.assistant.tools import ToolContext, _int, _opt_num, _p, _str, tool
from src.captions import styles

DEFAULT_HIGHLIGHT = {"color": "red", "scale": 1.25}  # what "highlight it" means when the user gave no colour or size


@tool("set_caption_style", "Change how the burned-in captions look. classic = white bold text, spoken word yellow. "
      "bold_pop = very large ALL CAPS, 1-3 words at a time, spoken word green and popping (Hormozi style). "
      "karaoke_box = the spoken word sits on a yellow highlight box. Applies to the whole video.",
      {"style": _p("string", "the caption style", list(styles.STYLES))}, required=("style",), mutating=True,
      label=lambda a: f"Setting the {a.get('style')} caption style")
def _set_caption_style(ctx: ToolContext, args: dict) -> dict:
    return {"ok": True, "style": workflow.set_caption_style(ctx.job, _str(args, "style"))}


@tool("highlight_words", "Make a word or phrase stand out in the captions: a different colour and/or bigger. Give "
      "the words as the user said them (it searches what the speaker says, and the English/Hinglish caption text if "
      "that was generated). With no colour and no size it uses red and a bit bigger. If the words occur several "
      "times it highlights every occurrence unless `occurrence` says otherwise; tell the user how many it found.",
      {"phrase": _p("string", "the word(s) to highlight"),
       "color": _p("string", "colour name (" + ", ".join(styles.NAMED_COLORS) + ") or #RRGGBB"),
       "size": _p("number", "size multiplier 1.0 to 1.8 (1.4 = clearly bigger); leave out to keep the size"),
       "occurrence": _p("string", "which matches: all (default), first, last, or nth (then give `which`)",
                        phrases.OCCURRENCES),
       "which": _p("integer", "for occurrence=nth: 1 = the first match")},
      required=("phrase",), mutating=True, label=lambda a: f"Highlighting '{a.get('phrase')}'")
def _highlight_words(ctx: ToolContext, args: dict) -> dict:
    phrase = _str(args, "phrase")
    color, size = args.get("color"), _opt_num(args, "size")
    if (color is None or str(color).strip() == "") and size is None:
        color, size = DEFAULT_HIGHLIGHT["color"], DEFAULT_HIGHLIGHT["scale"]
    found = phrases.search(ctx.job, phrase)
    total, hits = len(found), phrases.select(found, args)
    done = []
    for h in hits:
        entry = workflow.add_emphasis(ctx.job, h["start"], h["end"], color=color, scale=size, text=h["heard_as"])
        done.append({"id": entry["id"], "at": h["start"], "words": h["heard_as"], "context": h["context"]})
    return {"ok": True, "found": total, "highlighted": done, "color": color, "size": size}


@tool("clear_highlights", "Remove highlighted words from the captions: one by id (see get_plan captions.highlights) "
      "or all of them when no id is given.",
      {"highlight_id": _p("integer", "id of one highlight; leave out to clear them all")}, mutating=True,
      label=lambda a: "Clearing highlights")
def _clear_highlights(ctx: ToolContext, args: dict) -> dict:
    hid = None if args.get("highlight_id") is None else _int(args, "highlight_id")
    return {"ok": True, "removed": workflow.remove_emphasis(ctx.job, hid)}
