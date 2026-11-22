// Port of src/transcription/audio.py::extract_audio -- same flags, same 16kHz mono FLAC target
// (Whisper's native format), via ffmpeg_kit_flutter_new instead of a subprocess.

import 'dart:io';

import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/return_code.dart';

import '../utils/errors.dart';

Future<File> extractAudio(String videoPath, String outPath) async {
  if (!await File(videoPath).exists()) {
    throw AudioExtractionError('Video not found: $videoPath');
  }
  final cmd = '-nostdin -y -loglevel error -i "$videoPath" -vn -ac 1 -ar 16000 -c:a flac "$outPath"';
  final session = await FFmpegKit.execute(cmd);
  final rc = await session.getReturnCode();
  final out = File(outPath);
  if (!ReturnCode.isSuccess(rc) || !await out.exists()) {
    final logs = await session.getAllLogsAsString();
    throw AudioExtractionError('FFmpeg audio extraction failed: ${logs ?? rc}');
  }
  return out;
}
