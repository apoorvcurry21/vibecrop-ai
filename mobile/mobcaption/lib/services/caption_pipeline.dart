// Orchestrates the two pipeline stages around the caption editor:
//   1. transcribeVideo: audio extraction -> Groq transcription -> local chunking (feeds the
//      Edit Captions screen).
//   2. renderVideo: write the (possibly user-edited) chunks to an .ass file and burn it in.

import 'dart:io';

import '../captions/ass_writer.dart';
import '../captions/chunks.dart';
import '../captions/styles.dart';
import '../models/caption_chunk.dart';
import '../models/transcript.dart';
import '../utils/file_paths.dart';
import 'audio_extractor.dart';
import 'font_assets.dart';
import 'groq_transcriber.dart' as groq;
import 'video_probe.dart';
import 'video_renderer.dart';

enum PipelineStage { extractingAudio, transcribing, buildingCaptions }

class TranscribeResult {
  final Transcript transcript;
  final List<CaptionChunk> chunks;
  final VideoProbe probe;
  const TranscribeResult({required this.transcript, required this.chunks, required this.probe});
}

Future<TranscribeResult> transcribeVideo({
  required JobPaths job,
  required String videoPath,
  required String apiKey,
  required String language,
  void Function(PipelineStage stage)? onStage,
}) async {
  onStage?.call(PipelineStage.extractingAudio);
  await extractAudio(videoPath, job.audio);

  onStage?.call(PipelineStage.transcribing);
  final transcript = await groq.transcribe(audioFile: File(job.audio), apiKey: apiKey, language: language);

  onStage?.call(PipelineStage.buildingCaptions);
  const style = CaptionStyle.classic();
  final chunks = fitChunks(chunkWords(transcript.words), style.maxWords);
  final probe = await probeVideo(videoPath);

  return TranscribeResult(transcript: transcript, chunks: chunks, probe: probe);
}

Future<RenderResult> renderVideo({
  required JobPaths job,
  required String videoPath,
  required List<CaptionChunk> chunks,
  required VideoProbe probe,
  required CaptionStyle style,
  void Function(double fraction)? onProgress,
}) async {
  final events = captionEvents(chunks, style);
  await writeAss(job.overlayAss, events, style, playResX: probe.width, playResY: probe.height);
  final fontsDir = await ensureFontsDir(job.fontsDir);

  return burnCaptions(
    videoPath: videoPath,
    assPath: job.overlayAss,
    outputPath: job.output,
    probe: probe,
    fontsDir: fontsDir,
    onProgress: onProgress,
  );
}
