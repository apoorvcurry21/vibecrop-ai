// Golden-value port of tests/test_captions.py's chunking/retime cases, same inputs and
// expected outputs, to confirm the Dart port is behaviourally identical to the Python original.

import 'package:flutter_test/flutter_test.dart';
import 'package:mobcaption/captions/chunks.dart';
import 'package:mobcaption/models/word.dart';

List<TranscriptWord> words(List<(String, double, double)> spec) =>
    [for (final (w, s, e) in spec) TranscriptWord(word: w, start: s, end: e)];

String _rep(String s, int n) => List.filled(n, s).join();

void main() {
  test('chunks break on max words, a pause, and a sentence end', () {
    final ws = words([
      ('a', 0, .2),
      ('b', .2, .4),
      ('c', .4, .6),
      ('d', .6, .8),
      ('e', .8, 1.0),
      ('f', 3.0, 3.2),
      ('g.', 3.2, 3.4),
      ('h', 3.4, 3.6),
    ]);
    final chunks = chunkWords(ws, maxWords: 4);

    expect(
      [for (final c in chunks) [for (final w in c.words) w.text]],
      [
        ['a', 'b', 'c', 'd'],
        ['e'],
        ['f', 'g.'],
        ['h'],
      ],
    );
    expect(chunks[2].start, 3.0);
    expect(chunks[2].end, 3.4);
    expect([for (final c in chunks) c.id], [0, 1, 2, 3]);
  });

  test('chunks break on a Devanagari danda and on long text', () {
    final danda = chunkWords(words([('नमस्ते।', 0, .5), ('दोस्तों', .5, 1.0)]));
    expect(danda.length, 2);

    final long = chunkWords(
      words([(_rep('x', 14), 0, .3), (_rep('y', 14), .3, .6)]),
      maxChars: 26,
    );
    expect(long.length, 2);
  });

  test('retime reuses timings when the word count matches, else spreads', () {
    final orig = [
      const ChunkWord(text: 'a', start: 1.0, end: 1.5),
      const ChunkWord(text: 'b', start: 1.5, end: 2.0),
    ];

    final same = retime('Hello world', 1.0, 2.0, orig);
    expect([for (final w in same) (w.start, w.end)], [(1.0, 1.5), (1.5, 2.0)]);

    final spread = retime('one two three', 1.0, 2.0, orig);
    expect(spread.first.start, 1.0);
    expect(spread.last.end, closeTo(2.0, 0.01));
    for (var i = 0; i < spread.length - 1; i++) {
      expect(spread[i].end, lessThanOrEqualTo(spread[i + 1].start + 1e-6));
    }
  });

  test('retime carries emphasis over by index when the word count is unchanged, else resets', () {
    final orig = [
      const ChunkWord(text: 'a', start: 1.0, end: 1.5, emphasized: true),
      const ChunkWord(text: 'b', start: 1.5, end: 2.0),
    ];

    final same = retime('Hello world', 1.0, 2.0, orig);
    expect(same[0].emphasized, isTrue); // carried over -- same word count, same index
    expect(same[1].emphasized, isFalse);

    final spread = retime('one two three', 1.0, 2.0, orig);
    expect(spread.every((w) => !w.emphasized), isTrue); // word count changed -- reset, not fuzzy-matched
  });
}
