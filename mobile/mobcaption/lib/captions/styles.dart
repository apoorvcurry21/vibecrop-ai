// Port of src/captions/styles.py: the three caption presets (classic, bold_pop/"Hormozi",
// karaoke_box) plus fit_chunks. No colour/scale picker for emphasis in v1 -- see ass_writer.dart
// for the fixed emphasis scale, matching the user's "just size, keep it simple" ask.

import '../models/caption_chunk.dart';
import '../models/word.dart';

class CaptionStyle {
  final String id;
  final String label;
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
  final String? boxColor; // RGB hex, only set when box is true
  final int marginV;

  const CaptionStyle({
    required this.id,
    required this.label,
    required this.font,
    required this.size,
    required this.sizeDeva,
    required this.bold,
    required this.outline,
    required this.shadow,
    required this.uppercase,
    required this.maxWords,
    required this.primary,
    required this.active,
    required this.pop,
    required this.box,
    this.boxColor,
    required this.marginV,
  });

  const CaptionStyle.classic()
      : id = 'classic',
        label = 'Classic',
        font = 'Arial',
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
        boxColor = null,
        marginV = 400;

  const CaptionStyle.hormozi()
      : id = 'bold_pop',
        label = 'Hormozi',
        font = 'Anton',
        size = 104,
        sizeDeva = 84,
        bold = false,
        outline = 9,
        shadow = 3,
        uppercase = true,
        maxWords = 3,
        primary = 'FFFFFF',
        active = '39FF14',
        pop = true,
        box = false,
        boxColor = null,
        marginV = 400;

  const CaptionStyle.karaokeBox()
      : id = 'karaoke_box',
        label = 'Karaoke box',
        font = 'Arial',
        size = 78,
        sizeDeva = 72,
        bold = true,
        outline = 6,
        shadow = 0,
        uppercase = false,
        maxWords = 3,
        primary = 'FFFFFF',
        active = '000000',
        pop = false,
        box = true,
        boxColor = 'FFD60A',
        marginV = 400;

  static const List<CaptionStyle> all = [CaptionStyle.classic(), CaptionStyle.hormozi(), CaptionStyle.karaokeBox()];
}

/// Splits chunks longer than `maxWords` into balanced pieces (ceil division), same as
/// styles.py's fit_chunks. Lets switching to a smaller-maxWords style re-balance already
/// transcribed chunks without another Groq call.
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
