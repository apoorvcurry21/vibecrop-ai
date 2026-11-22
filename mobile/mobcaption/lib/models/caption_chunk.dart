import 'word.dart';

/// A group of 2-4 words shown as one caption line (port of chunks.py's chunk dict).
class CaptionChunk {
  final int id;
  final double start;
  final double end;
  final List<ChunkWord> words;

  const CaptionChunk({required this.id, required this.start, required this.end, required this.words});

  CaptionChunk copyWith({int? id, double? start, double? end, List<ChunkWord>? words}) => CaptionChunk(
        id: id ?? this.id,
        start: start ?? this.start,
        end: end ?? this.end,
        words: words ?? this.words,
      );
}
