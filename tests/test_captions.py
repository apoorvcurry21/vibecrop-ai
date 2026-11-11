import json
from types import SimpleNamespace

import pytest

from src.captions.chunks import chunk_words, retime
from src.captions.rewrite import CaptionError, rewrite_chunks
from src.rendering.ass import caption_events


def words(spec):
    return [{"word": w, "start": s, "end": e} for w, s, e in spec]


def test_chunks_break_on_max_words_pause_and_sentence_end():
    ws = words([("a", 0, .2), ("b", .2, .4), ("c", .4, .6), ("d", .6, .8), ("e", .8, 1.0),
                ("f", 3.0, 3.2), ("g.", 3.2, 3.4), ("h", 3.4, 3.6)])
    chunks = chunk_words(ws, max_words=4)
    assert [[w["text"] for w in c["words"]] for c in chunks] == [["a", "b", "c", "d"], ["e"], ["f", "g."], ["h"]]
    assert chunks[2]["start"] == 3.0 and chunks[2]["end"] == 3.4 and [c["id"] for c in chunks] == [0, 1, 2, 3]


def test_chunks_break_on_devanagari_danda_and_long_text():
    chunks = chunk_words(words([("नमस्ते।", 0, .5), ("दोस्तों", .5, 1.0)]))
    assert len(chunks) == 2
    long = chunk_words(words([("x" * 14, 0, .3), ("y" * 14, .3, .6)]), max_chars=26)
    assert len(long) == 2


def test_retime_reuses_timings_when_word_count_matches_else_spreads():
    orig = [{"text": "a", "start": 1.0, "end": 1.5}, {"text": "b", "start": 1.5, "end": 2.0}]
    same = retime("Hello world", 1.0, 2.0, orig)
    assert [(w["start"], w["end"]) for w in same] == [(1.0, 1.5), (1.5, 2.0)]
    spread = retime("one two three", 1.0, 2.0, orig)
    assert spread[0]["start"] == 1.0 and spread[-1]["end"] == pytest.approx(2.0, abs=0.01)
    assert all(a["end"] <= b["start"] + 1e-6 for a, b in zip(spread, spread[1:]))


class FakeClient:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.calls.append(contents)
        return SimpleNamespace(text=self.replies.pop(0))


CHUNKS = chunk_words(words([("बैलन", 0, .4), ("डियोर", .4, .9), ("जीता", 1.0, 1.4), ("था", 1.4, 1.6)]), max_words=2)


def reply(*texts):
    return json.dumps({"chunks": [{"id": i, "text": t} for i, t in enumerate(texts)]})


def test_original_and_english_to_english_are_noops():
    assert rewrite_chunks(CHUNKS, "original", "hi", client=object()) is CHUNKS
    assert rewrite_chunks(CHUNKS, "english", "en", client=object()) is CHUNKS


def test_hinglish_rewrite_keeps_ids_and_timings():
    client = FakeClient(reply("Ballon d'Or", "jeeta tha"))
    out = rewrite_chunks(CHUNKS, "hinglish", "hi", context="Messi", client=client, sleep=lambda s: None)
    assert [" ".join(w["text"] for w in c["words"]) for c in out] == ["Ballon d'Or", "jeeta tha"]
    assert (out[0]["start"], out[1]["end"]) == (CHUNKS[0]["start"], CHUNKS[1]["end"])
    assert out[1]["words"][0]["start"] == 1.0  # same word count -> original word timings reused
    assert "Messi" in client.calls[0][0].parts[0].text and "बैलन डियोर" in client.calls[0][0].parts[0].text


def test_bad_ids_trigger_retry_then_succeed_or_fail():
    ok = FakeClient(reply("only one"), reply("a b", "c d"))
    assert len(rewrite_chunks(CHUNKS, "english", "hi", client=ok, sleep=lambda s: None)) == 2
    assert len(ok.calls) == 2
    with pytest.raises(CaptionError):
        rewrite_chunks(CHUNKS, "english", "hi", client=FakeClient(*[reply("x")] * 3), sleep=lambda s: None)


def test_caption_events_highlight_active_word_and_pick_font_style():
    chunks = [
        {"id": 0, "start": 1.0, "end": 2.0, "words": [
            {"text": "hello", "start": 1.0, "end": 1.4}, {"text": "wor{ld}", "start": 1.4, "end": 2.0}]},
        {"id": 1, "start": 2.05, "end": 3.0, "words": [{"text": "नमस्ते", "start": 2.05, "end": 3.0}]},
    ]
    ev = caption_events(chunks)
    assert len(ev) == 3
    assert "CaptionLatin" in ev[0] and "{\\c&H0000FFFF&}hello{\\c&H00FFFFFF&} wor(ld)" in ev[0]
    assert "hello {\\c&H0000FFFF&}wor(ld)" in ev[1]
    assert "0:00:01.40,0:00:02.05" in ev[1]  # last word's end is clamped to the next chunk's start
    assert "CaptionDeva" in ev[2]
