// Golden-value port of src/rendering/ass.py's helpers and
// tests/test_captions.py::test_caption_events_highlight_active_word_and_pick_font_style.

import 'package:flutter_test/flutter_test.dart' hide escape;
import 'package:mobcaption/captions/ass_writer.dart';
import 'package:mobcaption/captions/styles.dart';
import 'package:mobcaption/models/caption_chunk.dart';
import 'package:mobcaption/models/word.dart';

void main() {
  test('bgr converts RGB hex to ASS BGR', () {
    expect(bgr('FF3B30'), '&H00303BFF&');
  });

  test('assTime formats seconds as H:MM:SS.CC', () {
    expect(assTime(3661.23), '1:01:01.23');
    expect(assTime(0), '0:00:00.00');
  });

  test('escape strips override-tag characters and collapses whitespace', () {
    expect(escape('a{b}\\c   d'), 'a(b)/c d');
  });

  test('caption events highlight the active word and pick the right font style', () {
    const style = CaptionStyle.classic();
    final chunks = [
      const CaptionChunk(id: 0, start: 1.0, end: 2.0, words: [
        ChunkWord(text: 'hello', start: 1.0, end: 1.4),
        ChunkWord(text: 'wor{ld}', start: 1.4, end: 2.0),
      ]),
      const CaptionChunk(id: 1, start: 2.05, end: 3.0, words: [
        ChunkWord(text: 'नमस्ते', start: 2.05, end: 3.0),
      ]),
    ];

    final ev = captionEvents(chunks, style);

    expect(ev.length, 3);
    expect(ev[0], contains('CaptionLatin'));
    expect(ev[0], contains(r'{\c&H0000FFFF&}hello{\c&H00FFFFFF&} wor(ld)'));
    expect(ev[1], contains(r'hello {\c&H0000FFFF&}wor(ld)'));
    expect(ev[1], contains('0:00:01.40,0:00:02.05')); // last word's end clamped to next chunk's start
    expect(ev[2], contains('CaptionDeva'));
  });
}
