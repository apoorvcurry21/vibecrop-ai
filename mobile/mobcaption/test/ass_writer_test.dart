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

  // Chunk matching tests/test_caption_styles.py's CHUNK ("You will never quit", 10.0-11.6),
  // reused below for golden-value parity with the desktop app's ass.py tests.
  const chunk = CaptionChunk(id: 0, start: 10.0, end: 11.6, words: [
    ChunkWord(text: 'You', start: 10.0, end: 10.3),
    ChunkWord(text: 'will', start: 10.4, end: 10.7),
    ChunkWord(text: 'never', start: 10.8, end: 11.2),
    ChunkWord(text: 'quit', start: 11.2, end: 11.6),
  ]);

  test('hormozi style is uppercase, green, and pops (test_bold_pop_is_uppercase_green_and_pops)', () {
    const style = CaptionStyle.hormozi();
    final line = captionEvents([chunk], style)[1]; // i=1 -> "will" is the active word
    expect(line, contains('WILL'));
    expect(line, isNot(contains('will')));
    expect(line, contains(r'\c&H0014FF39&')); // 39FF14 in BGR
    expect(line, contains(r'\fscx82\fscy82\t(0,90,\fscx100\fscy100)'));
    // Deliberate substitute for Impact (not available on Android) -- see styles.dart.
    expect(
      header(style, playResX: 1080, playResY: 1920),
      contains('Style: CaptionLatin,Anton,104'),
    );
  });

  test('karaoke box adds a second layer, only the spoken word visible in it '
      '(test_karaoke_box_adds_a_second_layer_only_the_spoken_word_shows_in)', () {
    const style = CaptionStyle.karaokeBox();
    final ev = captionEvents([chunk], style);
    expect(ev.length, 8); // base + box per word, 4 words
    final base = ev[2], box = ev[3]; // i=1 -> "will"
    expect(base, startsWith('Dialogue: 1,'));
    expect(base, contains(',CaptionLatin,'));
    expect(base, isNot(contains(r'\c&H')));
    expect(box, startsWith('Dialogue: 2,'));
    expect(box, contains(',ActiveLatin,'));
    expect(r'\alpha&HFF&'.allMatches(box).length, 3);
    expect(box, contains('never'));
    final h = header(style, playResX: 1080, playResY: 1920);
    expect(h, contains('Style: ActiveLatin,'));
    expect(h, contains('Style: ActiveDeva,'));
  });

  test(
      'emphasis adds size and both layers agree on it '
      '(test_emphasis_adds_colour_and_size_and_both_layers_agree_on_size, size-only variant)', () {
    final emphasizedChunk = chunk.copyWith(words: [
      chunk.words[0],
      chunk.words[1],
      chunk.words[2].copyWith(emphasized: true), // "never"
      chunk.words[3],
    ]);

    // Active AND emphasized ("never" is both the spoken word and marked bigger): colour + size combine.
    const classic = CaptionStyle.classic();
    final activeAndEmphasized = captionEvents([emphasizedChunk], classic)[2]; // i=2 -> "never" active
    expect(activeAndEmphasized, contains(r'{\c&H0000FFFF&\fscx135\fscy135}never{\r}'));

    // Emphasized but NOT the active word: size tag only, no colour tag.
    final sizeOnly = captionEvents([emphasizedChunk], classic)[0]; // i=0 -> "You" active, "never" just emphasized
    expect(sizeOnly, contains(r'{\fscx135\fscy135}never{\r}'));

    // karaoke_box: both the base line and the box line size "never" identically, so the box still
    // lines up with the word even though this event's active word is "You", not "never".
    const box = CaptionStyle.karaokeBox();
    final boxEvents = captionEvents([emphasizedChunk], box);
    expect(boxEvents[0], contains(r'\fscx135\fscy135}never')); // base layer
    expect(boxEvents[1], contains(r'\alpha&HFF&\fscx135\fscy135}never')); // box layer, still invisible
  });
}
