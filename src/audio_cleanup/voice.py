"""The voice-polish chain applied to the cleaned audio: rumble filter, a gentle EQ, a compressor, loudness.

Every value is a named constant so it is easy to tune. They are conservative on purpose: the goal is a clear,
even voice, not a "radio" sound. The chain runs AFTER the cuts, so the compressor never reacts to removed parts.
"""
from __future__ import annotations

HIGHPASS_HZ = 80  # removes desk thumps and rumble below the voice
MUD_HZ, MUD_GAIN_DB = 250, -2.0  # a little less "boxiness"
PRESENCE_HZ, PRESENCE_GAIN_DB = 3000, 2.5  # a little more clarity / consonants
EQ_Q = 1.0
COMP_THRESHOLD = 0.1  # linear amplitude (about -20 dB): only the louder parts are pressed down
COMP_RATIO = 3.0
COMP_ATTACK_MS, COMP_RELEASE_MS = 5, 80
COMP_MAKEUP = 1.5  # linear gain after compression (loudnorm sets the final level anyway)
TARGET_LUFS = -16.0  # the usual level for short-form video; keeps added sound effects balanced against the voice
TRUE_PEAK_DB = -1.5
LOUDNESS_RANGE = 11
OUTPUT_RATE = 48000  # loudnorm works internally at 192 kHz; bring it back to a normal rate


def voice_filter() -> str:
    return (f"highpass=f={HIGHPASS_HZ},"
            f"equalizer=f={MUD_HZ}:t=q:w={EQ_Q}:g={MUD_GAIN_DB},"
            f"equalizer=f={PRESENCE_HZ}:t=q:w={EQ_Q}:g={PRESENCE_GAIN_DB},"
            f"acompressor=threshold={COMP_THRESHOLD}:ratio={COMP_RATIO}:attack={COMP_ATTACK_MS}:"
            f"release={COMP_RELEASE_MS}:makeup={COMP_MAKEUP},"
            f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK_DB}:LRA={LOUDNESS_RANGE},"
            f"aresample={OUTPUT_RATE}")
