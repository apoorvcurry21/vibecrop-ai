"""Measure the chat assistant against the live model: a fixed list of English and Hinglish commands, each run on a
fresh copy of a small demo project, then a check of what really changed in plan.json.

    python scripts/eval_assistant.py [--only NAME_PART ...] [--out results.json] [--pause 2]

Needs GEMINI_API_KEY (it uses the free-tier quota, roughly 2-4 model calls per command). Nothing here touches your
real jobs: each case builds its own project in a temp folder. Network-only commands (YouTube search, sound
download) are left out on purpose, so the run never downloads anything."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import workflow  # noqa: E402
from src.assistant import agent, tools  # noqa: E402
from tests.test_chat_agent import make_job  # noqa: E402  (the same demo project the unit tests use)


def slot(plan, i):
    return next((x for x in plan["items"] if x["id"] == i), None)


def emph(plan):
    return plan.get("captions", {}).get("emphasis", [])


def near(a, b, tol=0.6):
    return a is not None and abs(float(a) - b) <= tol


def pip_first(job: Path) -> None:
    """Setup: slot 1 is already a small box in the top-right corner."""
    from src.assistant.tools import ToolContext, run_tool
    run_tool(ToolContext(job=job), "set_pip", {"slot_id": 1, "position": "top-right"})


def split_first(job: Path) -> None:
    from src.assistant.tools import ToolContext, run_tool
    run_tool(ToolContext(job=job), "set_layout", {"slot_id": 1, "layout": "split"})


# name, what the user types, check(plan, calls, reply) -> bool.  Demo project: slots 1 (2-6 s) and 2 (14-18 s), both
# overlay; the speaker says "virat koli is the ronaldo of cricket" (virat 7.0, ronaldo 8.3, cricket 9.2).
CASES = [
    ("split_en", "make slot 2 split", lambda p, c, r: slot(p, 2)["layout"] == "split" and slot(p, 1)["layout"] == "overlay"),
    ("split_hi", "slot 1 ko split kardo", lambda p, c, r: slot(p, 1)["layout"] == "split" and slot(p, 2)["layout"] == "overlay"),
    ("pip_hi", "doosre broll ko chhote box mein daalo, top right",
     lambda p, c, r: slot(p, 2)["layout"] == "pip" and slot(p, 2)["pip"]["pos"] == "top-right" and slot(p, 1)["layout"] == "overlay"),
    ("remove_en", "remove the first broll", lambda p, c, r: slot(p, 1) is not None and slot(p, 1).get("enabled") is False and slot(p, 2).get("enabled") is not False),
    ("remove_hi", "pehla broll hata do", lambda p, c, r: slot(p, 1) is not None and slot(p, 1).get("enabled") is False and slot(p, 2).get("enabled") is not False),
    ("delete_hi", "slot 2 delete kardo permanently", lambda p, c, r: slot(p, 2) is None and slot(p, 1) is not None),
    ("move_en", "move slot 2 so it starts at 10 seconds", lambda p, c, r: near(slot(p, 2)["timestamp_start"], 10.0, 0.2)),
    ("longer_en", "make slot 1 last until 8 seconds", lambda p, c, r: near(slot(p, 1)["timestamp_end"], 8.0, 0.2) and near(slot(p, 1)["timestamp_start"], 2.0, 0.2)),
    ("swap_en", "swap the clips of slot 1 and slot 2",
     lambda p, c, r: slot(p, 1)["asset"]["title"] == "Title 2" and slot(p, 2)["asset"]["title"] == "Title 1"),
    ("style_bold", "make the captions bold pop", lambda p, c, r: p.get("captions", {}).get("style") == "bold_pop"),
    ("style_hi", "caption ko karaoke box style mein karo", lambda p, c, r: p.get("captions", {}).get("style") == "karaoke_box"),
    ("hl_red", "highlight the word cricket in red",
     lambda p, c, r: len(emph(p)) == 1 and near(emph(p)[0]["start"], 9.2, 0.3) and emph(p)[0]["color"] == "FF3B30"),
    ("hl_big_hi", "cricket word ko bada karo, rang mat badalna",
     lambda p, c, r: len(emph(p)) == 1 and near(emph(p)[0]["start"], 9.2, 0.3) and emph(p)[0]["color"] is None and emph(p)[0]["scale"] > 1.0),
    ("text_en", "add the text GOAT at 5 seconds for 2 seconds",
     lambda p, c, r: any("GOAT" in t["text"].upper() and near(t["start"], 5.0, 0.3) for t in p.get("texts", []))),
    ("text_on_word", "put the text 'wait for it' when I say ronaldo",
     lambda p, c, r: any("wait" in t["text"].lower() and near(t["start"], 8.3, 0.8) for t in p.get("texts", []))),
    ("zoom_in_en", "slowly zoom in on slot 1",
     lambda p, c, r: slot(p, 1).get("framing") is not None and slot(p, 1)["framing"]["end"]["z"] > slot(p, 1)["framing"]["start"]["z"]),
    ("nozoom_hi", "slot 1 pe zoom bilkul mat karo, poora clip dikhao",
     lambda p, c, r: slot(p, 1).get("framing") is None or (slot(p, 1)["framing"]["start"]["z"] == slot(p, 1)["framing"]["end"]["z"] == 1.0 and slot(p, 1)["framing"].get("base", "fit") == "fit")),
    ("all_split", "make every broll split", lambda p, c, r: all(i["layout"] == "split" for i in p["items"])),
    ("only_first_pip", "sirf pehle broll ko pip bana do, dusre ko mat chhedna",
     lambda p, c, r: slot(p, 1)["layout"] == "pip" and slot(p, 2)["layout"] == "overlay"),
    # --- harder ones, taken from how the real chats went wrong: several asks in one message, relative moves, "middle" meaning
    # the middle of the SCREEN, "last" / "both", and undo.
    ("multi_hi", "slot 1 ko split karo, cricket word red karo, aur captions bold pop kardo",
     lambda p, c, r: slot(p, 1)["layout"] == "split" and p.get("captions", {}).get("style") == "bold_pop"
     and len(emph(p)) == 1 and emph(p)[0]["color"] == "FF3B30"),
    ("earlier_hi", "slot 2 ko 3 second pehle le aao", lambda p, c, r: near(slot(p, 2)["timestamp_start"], 11.0, 0.3) and near(slot(p, 2)["timestamp_end"], 15.0, 0.3)),
    ("last_hi", "last wala broll hata do", lambda p, c, r: slot(p, 2).get("enabled") is False and slot(p, 1).get("enabled") is not False),
    ("both_pip", "dono broll ko small box mein bottom left pe rakho",
     lambda p, c, r: all(i["layout"] == "pip" and i["pip"]["pos"] == "bottom-left" for i in p["items"])),
    ("middle_screen", "slot 1 ko screen ke beech mein chhote box mein rakho",
     lambda p, c, r: slot(p, 1)["layout"] == "pip" and slot(p, 1)["pip"]["pos"] == "center"),
    ("middle_hi", "slot 1 wala picture middle mai le aa",
     lambda p, c, r: slot(p, 1)["layout"] == "pip" and slot(p, 1)["pip"]["pos"] == "center" and near(slot(p, 1)["timestamp_start"], 2.0, 0.1), pip_first),
    ("middle_of_video", "slot 1 ko video ke beech mein shift kar do (timeline mein)",
     lambda p, c, r: 11.0 <= slot(p, 1)["timestamp_start"] <= 14.0),
    ("text_at_zero", "add the text HELLO from the very start 0 to 2.4 seconds",
     lambda p, c, r: any("HELLO" in t["text"].upper() and near(t["start"], 0.0, 0.1) and near(t["end"], 2.4, 0.2) for t in p.get("texts", []))),
    ("undo_en", "undo that", lambda p, c, r: slot(p, 1)["layout"] == "overlay", split_first),
    ("honest_noop", "make slot 1 split",
     lambda p, c, r: slot(p, 1)["layout"] == "split" and not any(w in r.lower() for w in ("changed it", "i changed", "i have changed")), split_first),
    # No change expected: a question, and a slot that does not exist.
    ("question", "what is in slot 2?", lambda p, c, r: not any(tools.REGISTRY[n].mutating for n, _ in c if n in tools.REGISTRY)),
    ("no_such_slot", "slot 5 ko split karo",
     lambda p, c, r: all(i["layout"] == "overlay" for i in p["items"]) and len(p["items"]) == 2),
]


def run_case(name: str, text: str, check, setup=None) -> dict:  # noqa: C901
    with tempfile.TemporaryDirectory() as d:
        job = make_job(Path(d))
        if setup:
            setup(job)
        calls: list[tuple[str, dict]] = []
        model_calls = [0]
        real_run, real_open = tools.run_tool, agent.open_stream

        def run_tool(ctx, tool_name, args):
            calls.append((tool_name, dict(args)))
            return real_run(ctx, tool_name, args)

        def open_stream(*a, **k):
            model_calls[0] += 1
            return real_open(*a, **k)

        tools.run_tool, agent.open_stream = run_tool, open_stream
        reply, error = "", ""
        t0 = time.time()
        try:
            for ev in agent.stream_turn(job, [], text):
                if ev["type"] == "error":
                    error = ev["text"]
                elif ev["type"] == "done":
                    reply = ev["text"]
        finally:
            tools.run_tool, agent.open_stream = real_run, real_open
        took = time.time() - t0
        try:
            ok = bool(check(workflow.load_plan(job), calls, reply)) and not error
        except Exception as exc:  # noqa: BLE001 - a check that cannot even read the state is a failed case
            ok, error = False, f"check crashed: {exc!r}"
        return {"case": name, "text": text, "ok": ok, "seconds": round(took, 1), "model_calls": model_calls[0],
                "tools": [f"{n}({json.dumps(a, ensure_ascii=False)})" for n, a in calls], "reply": reply[:300],
                "error": error}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[], help="run only cases whose name contains one of these")
    ap.add_argument("--out", default="", help="write the full results as JSON to this file")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between cases (free-tier rate limits)")
    args = ap.parse_args()
    results = []
    for name, text, check, *setup in CASES:
        if args.only and not any(o in name for o in args.only):
            continue
        r = run_case(name, text, check, *setup)
        results.append(r)
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {name:14} {r['seconds']:5.1f}s  {r['model_calls']} model calls  "
              f"{' | '.join(r['tools']) or '(no tool)'}" + (f"   ERROR: {r['error']}" if r["error"] else ""), flush=True)
        time.sleep(args.pause)
    if not results:
        return
    passed = sum(r["ok"] for r in results)
    print(f"\n{passed}/{len(results)} passed; median {statistics.median(r['seconds'] for r in results):.1f}s per command, "
          f"mean {statistics.mean(r['model_calls'] for r in results):.1f} model calls per command")
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
