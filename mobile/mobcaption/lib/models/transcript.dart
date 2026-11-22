import 'word.dart';

class TranscriptSegment {
  final int id;
  final double start;
  final double end;
  final String text;

  const TranscriptSegment({required this.id, required this.start, required this.end, required this.text});
}

/// Port of groq_whisper.py's normalize() output shape.
class Transcript {
  final String language;
  final String model;
  final double duration;
  final String text;
  final List<TranscriptSegment> segments;
  final List<TranscriptWord> words;

  const Transcript({
    required this.language,
    required this.model,
    required this.duration,
    required this.text,
    required this.segments,
    required this.words,
  });
}
