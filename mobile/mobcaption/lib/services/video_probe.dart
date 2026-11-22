// Probes the source video's own resolution/fps/bitrate via FFprobeKit, so the burn-in render
// never forces a fixed output size and never degrades quality below what the source already has.

import 'package:ffmpeg_kit_flutter_new/ffprobe_kit.dart';
import 'package:ffmpeg_kit_flutter_new/stream_information.dart';

import '../utils/errors.dart';

class VideoProbe {
  final int width;
  final int height;
  final double fps;
  final int bitrateBps; // 0 if ffprobe couldn't report one
  final double durationSeconds;

  const VideoProbe({
    required this.width,
    required this.height,
    required this.fps,
    required this.bitrateBps,
    required this.durationSeconds,
  });
}

Future<VideoProbe> probeVideo(String path) async {
  final session = await FFprobeKit.getMediaInformation(path);
  final info = session.getMediaInformation();
  if (info == null) {
    throw const RenderError('Could not read the video — is the file a valid video?');
  }

  StreamInformation? videoStream;
  for (final s in info.getStreams()) {
    if (s.getType() == 'video') {
      videoStream = s;
      break;
    }
  }
  if (videoStream == null) {
    throw const RenderError('No video stream found in the selected file.');
  }

  final fps = _parseFraction(videoStream.getAverageFrameRate()) ?? _parseFraction(videoStream.getRealFrameRate()) ?? 30.0;
  final streamBitrate = int.tryParse(videoStream.getBitrate() ?? '');
  final formatBitrate = int.tryParse(info.getBitrate() ?? '');

  return VideoProbe(
    width: videoStream.getWidth() ?? 1080,
    height: videoStream.getHeight() ?? 1920,
    fps: fps,
    bitrateBps: streamBitrate ?? formatBitrate ?? 0,
    durationSeconds: double.tryParse(info.getDuration() ?? '') ?? 0.0,
  );
}

double? _parseFraction(String? s) {
  if (s == null || s.isEmpty) return null;
  final parts = s.split('/');
  if (parts.length != 2) return double.tryParse(s);
  final num = double.tryParse(parts[0]);
  final den = double.tryParse(parts[1]);
  if (num == null || den == null || den == 0) return null;
  final value = num / den;
  return value.isFinite && value > 0 ? value : null;
}
