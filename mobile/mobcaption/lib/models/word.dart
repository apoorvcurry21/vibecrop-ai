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

  const ChunkWord({required this.text, required this.start, required this.end});
}
