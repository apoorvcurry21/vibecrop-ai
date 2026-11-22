/// A single word as Groq's Whisper returns it: raw text (may carry punctuation) + timestamps.
class TranscriptWord {
  final String word;
  final double start;
  final double end;

  const TranscriptWord({required this.word, required this.start, required this.end});
}

/// A word inside a caption chunk: stripped text, ready to render.
class ChunkWord {
  final String text;
  final double start;
  final double end;
  final bool emphasized;

  const ChunkWord({required this.text, required this.start, required this.end, this.emphasized = false});

  ChunkWord copyWith({String? text, double? start, double? end, bool? emphasized}) => ChunkWord(
        text: text ?? this.text,
        start: start ?? this.start,
        end: end ?? this.end,
        emphasized: emphasized ?? this.emphasized,
      );
}
