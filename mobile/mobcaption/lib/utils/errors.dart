/// Mirrors src/transcription/groq_whisper.py's TranscriptionError.
class TranscriptionError implements Exception {
  final String message;
  const TranscriptionError(this.message);
  @override
  String toString() => message;
}

/// Mirrors src/transcription/audio.py's AudioExtractionError.
class AudioExtractionError implements Exception {
  final String message;
  const AudioExtractionError(this.message);
  @override
  String toString() => message;
}

/// Raised when the burn-in render (ffmpeg subtitles= pass) fails.
class RenderError implements Exception {
  final String message;
  const RenderError(this.message);
  @override
  String toString() => message;
}
