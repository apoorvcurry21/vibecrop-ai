// Port of src/captions/styles.py, trimmed to what v1 needs: only the "classic" preset (no style
// picker, no emphasis/highlight editing), plus fit_chunks for correctness.

import '../models/caption_chunk.dart';
import '../models/word.dart';

class CaptionStyle {
  final String font;
  final int size;
  final int sizeDeva;
  final bool bold;
  final double outline;
  final double shadow;
  final bool uppercase;
  final int maxWords;
  final String primary; // RGB hex, e.g. "FFFFFF"
  final String active; // RGB hex, e.g. "FFFF00"
  final bool pop;
  final bool box;
  final int marginV;

  const CaptionStyle.classic()
      : font = 'Arial',
        size = 74,
        sizeDeva = 70,
        bold = true,
        outline = 7,
        shadow = 2,
        uppercase = false,
        maxWords = 4,
        primary = 'FFFFFF',
        active = 'FFFF00',
        pop = false,
        box = false,
        marginV = 400;
}

/// Splits chunks longer than `maxWords` into balanced pieces (ceil division), same as
/// styles.py's fit_chunks. A no-op in v1 since chunkWords already caps at 4 == classic.maxWords,
/// but kept so a future style with a smaller maxWords works without a second pass over the words.
List<CaptionChunk> fitChunks(List<CaptionChunk> chunks, int maxWords) {
  final out = <List<ChunkWord>>[];
  for (final chunk in chunks) {
    final words = chunk.words;
    if (words.length <= maxWords) {
      out.add(words);
      continue;
    }
    final parts = (words.length / maxWords).ceil();
    final size = words.length ~/ parts;
    final extra = words.length % parts;
    var i = 0;
    for (var p = 0; p < parts; p++) {
      final n = size + (p < extra ? 1 : 0);
      out.add(words.sublist(i, i + n));
      i += n;
    }
  }
  return [
    for (var n = 0; n < out.length; n++)
      CaptionChunk(id: n, start: out[n].first.start, end: out[n].last.end, words: out[n]),
  ];
}
