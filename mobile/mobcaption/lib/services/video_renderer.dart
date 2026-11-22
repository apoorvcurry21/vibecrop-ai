// Burns the generated .ass captions onto the source video. Quality-preserving: never forces a
// fixed output resolution or a fps above the source's own, and floors the video bitrate at
// 25 Mbps (or the source's own bitrate if it's already higher). Tries the hardware encoder
// (h264_mediacodec) first for speed, falling back to software libx264 if it's unavailable or
// fails -- confirmed necessary by the Milestone 1b smoke test, where mediacodec failed on the
// test device and libx264 handled a 5s 720p clip in ~6.4s.

import 'dart:async';
import 'dart:io';

import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/return_code.dart';
import 'package:ffmpeg_kit_flutter_new/statistics.dart';

import '../utils/errors.dart';
import 'video_probe.dart';

const int qualityFloorBps = 25 * 1000 * 1000; // 25 Mbps floor

class RenderResult {
  final String outputPath;
  final String encoderUsed;
  const RenderResult({required this.outputPath, required this.encoderUsed});
}

Future<RenderResult> burnCaptions({
  required String videoPath,
  required String assPath,
  required String outputPath,
  required VideoProbe probe,
  required String fontsDir,
  void Function(double fractionComplete)? onProgress,
}) async {
  final targetBitrate = probe.bitrateBps > qualityFloorBps ? probe.bitrateBps : qualityFloorBps;
  // Never fabricate frames the source doesn't have, and never force a resize -- the output
  // frame stays exactly the source's own width/height via the subtitles filter alone.
  final fps = probe.fps > 0 ? probe.fps : 30.0;
  // fontsdir lets libass find the bundled Hormozi-style font without it being a system font.
  final vf = 'subtitles=$assPath:fontsdir=$fontsDir';

  Future<bool> attempt(String codecArgs) async {
    final out = File(outputPath);
    if (await out.exists()) await out.delete();

    final cmd = '-y -i "$videoPath" -vf "$vf" -c:a copy $codecArgs -r $fps -pix_fmt yuv420p "$outputPath"';
    final completer = Completer<ReturnCode?>();
    await FFmpegKit.executeAsync(cmd, (session) async {
      completer.complete(await session.getReturnCode());
    }, null, (Statistics stats) {
      if (onProgress != null && probe.durationSeconds > 0) {
        final processedSeconds = stats.getTime() / 1000.0;
        onProgress((processedSeconds / probe.durationSeconds).clamp(0.0, 1.0));
      }
    });
    final rc = await completer.future;
    return ReturnCode.isSuccess(rc) && await out.exists();
  }

  final hwArgs = '-c:v h264_mediacodec -b:v $targetBitrate -maxrate $targetBitrate -bufsize ${targetBitrate * 2}';
  if (await attempt(hwArgs)) {
    return RenderResult(outputPath: outputPath, encoderUsed: 'h264_mediacodec (hardware)');
  }

  final swArgs =
      '-c:v libx264 -preset veryfast -b:v $targetBitrate -maxrate $targetBitrate -bufsize ${targetBitrate * 2}';
  if (await attempt(swArgs)) {
    return RenderResult(outputPath: outputPath, encoderUsed: 'libx264 (software)');
  }

  throw const RenderError('Both hardware and software encoding failed. See the FFmpeg logs for details.');
}
