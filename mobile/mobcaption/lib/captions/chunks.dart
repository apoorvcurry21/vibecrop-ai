// Port of src/captions/chunks.py: group Whisper word timestamps into short caption chunks
// (2-4 words) and re-time edited/custom text onto a chunk's original time span.

import '../models/caption_chunk.dart';
import '../models/word.dart';

const List<String> sentenceEnd = ['.', '?', '!', '।', '…']; // includes the Devanagari danda

/// Groups `words` into chunks, breaking at pauses, sentence ends, or the word/char limits.
List<CaptionChunk> chunkWords(
  List<TranscriptWord> words, {
  int maxWords = 4,
  int maxChars = 26,
  double maxGap = 0.6,
}) {
  final chunks = <List<ChunkWord>>[];
  var cur = <ChunkWord>[];

  void flush() {
    if (cur.isNotEmpty) {
      chunks.add(cur);
      cur = <ChunkWord>[];
    }
  }

  for (final w in words) {
    final text = w.word.trim();
    if (text.isEmpty) continue;
    if (cur.isNotEmpty) {
      final gap = w.start - cur.last.end;
      final chars = cur.fold<int>(0, (sum, x) => sum + x.text.length + 1) + text.length;
      if (cur.length >= maxWords || gap > maxGap || chars > maxChars) {
        flush();
      }
    }
    cur.add(ChunkWord(text: text, start: w.start, end: w.end));
    if (sentenceEnd.any(text.endsWith)) {
      flush();
    }
  }
  flush();

  return [
    for (var i = 0; i < chunks.length; i++)
      CaptionChunk(id: i, start: chunks[i].first.start, end: chunks[i].last.end, words: chunks[i]),
  ];
}

/// Gives edited/custom caption text word timings: reuses the originals if the word count still
/// matches, otherwise spreads the words across [start, end] in proportion to their length.
/// Used by the caption editor: editing a chunk's text, or adding a brand-new caption, both need
/// per-word timestamps for the highlighted-word ASS markup even though there's no fresh Whisper
/// timing for the new words.
List<ChunkWord> retime(String text, double start, double end, List<ChunkWord> originalWords) {
  final tokens = text.split(RegExp(r'\s+')).where((t) => t.isNotEmpty).toList();
  if (tokens.isEmpty) return [];
  if (tokens.length == originalWords.length) {
    return [
      for (var i = 0; i < tokens.length; i++)
        ChunkWord(text: tokens[i], start: originalWords[i].start, end: originalWords[i].end),
    ];
  }
  final total = tokens.fold<int>(0, (sum, t) => sum + t.length + 1);
  final out = <ChunkWord>[];
  var cursor = start;
  for (final t in tokens) {
    final span = (end - start) * (t.length + 1) / total;
    final wStart = _round3(cursor);
    final wEnd = _round3(cursor + span);
    out.add(ChunkWord(text: t, start: wStart, end: wEnd));
    cursor += span;
  }
  return out;
}

double _round3(double x) => double.parse(x.toStringAsFixed(3));
