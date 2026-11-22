// mobcaption Milestone 1 + 1b smoke test.
//
// Proves two things the real app depends on, on a REAL device (not an emulator):
//  1. ffmpeg_kit_flutter_new's bundled ffmpeg can burn an .ass file onto video via the
//     `subtitles=` filter (libass), including the per-word color-swap markup the real
//     caption engine uses ({\c&H..&}word{\c&H..&}).
//  2. Whether a hardware encoder (h264_mediacodec) is actually usable for the quality
//     target (25 Mbps) on this device, falling back to software libx264 if not.
//
// This is throwaway code: not part of the real mobcaption app.

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/return_code.dart';
import 'package:path_provider/path_provider.dart';
import 'package:video_player/video_player.dart';

void main() => runApp(const SmokeTestApp());

class SmokeTestApp extends StatelessWidget {
  const SmokeTestApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'mobcaption smoke test',
      home: SmokeTestPage(),
    );
  }
}

enum _Stage { idle, preparing, encodingHw, encodingSw, done, failed }

class SmokeTestPage extends StatefulWidget {
  const SmokeTestPage({super.key});

  @override
  State<SmokeTestPage> createState() => _SmokeTestPageState();
}

class _SmokeTestPageState extends State<SmokeTestPage> {
  _Stage _stage = _Stage.idle;
  String _log = '';
  String? _outputPath;
  String? _encoderUsed;
  Duration? _elapsed;
  VideoPlayerController? _player;

  void _append(String line) {
    setState(() => _log = '$_log\n$line'.trim());
  }

  Future<File> _extractAsset(String assetPath, String fileName) async {
    final bytes = await DefaultAssetBundle.of(context).load(assetPath);
    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/$fileName');
    await file.writeAsBytes(bytes.buffer.asUint8List(bytes.offsetInBytes, bytes.lengthInBytes));
    return file;
  }

  // Tries the hardware encoder first (fast on real devices); falls back to libx264 if the
  // session fails. Returns the encoder name that actually worked, or null if both failed.
  Future<String?> _tryEncoders(String videoPath, String assPath, String outputPath) async {
    // The .ass path must have no ':' or special chars for ffmpeg's filter-string parser;
    // the temp dir path on Android is safe (no drive letters), unlike Windows.
    final vf = 'subtitles=$assPath';

    final hwCmd = '-y -i "$videoPath" -vf "$vf" -c:a copy -c:v h264_mediacodec '
        '-b:v 25M -maxrate 25M -bufsize 50M -pix_fmt yuv420p "$outputPath"';
    _append('Trying hardware encoder (h264_mediacodec)...');
    var session = await FFmpegKit.execute(hwCmd);
    var rc = await session.getReturnCode();
    if (ReturnCode.isSuccess(rc)) {
      return 'h264_mediacodec (hardware)';
    }
    _append('Hardware encoder failed (rc=$rc), falling back to libx264...');
    final logs = await session.getAllLogsAsString();
    _append(logs ?? '(no logs)');

    final swCmd = '-y -i "$videoPath" -vf "$vf" -c:a copy -c:v libx264 -preset veryfast '
        '-b:v 25M -maxrate 25M -bufsize 50M -pix_fmt yuv420p "$outputPath"';
    session = await FFmpegKit.execute(swCmd);
    rc = await session.getReturnCode();
    if (ReturnCode.isSuccess(rc)) {
      return 'libx264 (software)';
    }
    _append('Software encoder also failed (rc=$rc).');
    _append(await session.getAllLogsAsString() ?? '(no logs)');
    return null;
  }

  Future<void> _run() async {
    setState(() {
      _stage = _Stage.preparing;
      _log = '';
      _outputPath = null;
      _encoderUsed = null;
      _elapsed = null;
    });
    _player?.dispose();
    _player = null;

    try {
      _append('Extracting bundled test assets...');
      final video = await _extractAsset('assets/test_video.mp4', 'test_video.mp4');
      final ass = await _extractAsset('assets/test_overlay.ass', 'test_overlay.ass');
      final dir = await getTemporaryDirectory();
      final outputPath = '${dir.path}/smoke_output.mp4';
      // Delete any previous output so a stale file never gets mistaken for a fresh success.
      final existing = File(outputPath);
      if (await existing.exists()) await existing.delete();

      setState(() => _stage = _Stage.encodingHw);
      final stopwatch = Stopwatch()..start();
      final encoder = await _tryEncoders(video.path, ass.path, outputPath);
      stopwatch.stop();

      if (encoder == null || !await File(outputPath).exists()) {
        setState(() => _stage = _Stage.failed);
        return;
      }

      _append('Success with $encoder in ${stopwatch.elapsed.inMilliseconds} ms.');
      _player = VideoPlayerController.file(File(outputPath));
      await _player!.initialize();
      await _player!.setLooping(true);
      await _player!.play();

      setState(() {
        _stage = _Stage.done;
        _outputPath = outputPath;
        _encoderUsed = encoder;
        _elapsed = stopwatch.elapsed;
      });
    } catch (e, st) {
      _append('Exception: $e\n$st');
      setState(() => _stage = _Stage.failed);
    }
  }

  @override
  void dispose() {
    _player?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('mobcaption: libass smoke test')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Watch the preview below once it finishes: the word "HELLO" then '
              '"MOBCAPTION" should turn yellow as it is spoken, around the 0.5s and 2s '
              'marks. If you see plain white text with no color change, libass highlighting '
              'did not work and we need the drawtext fallback.',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: _stage == _Stage.encodingHw || _stage == _Stage.encodingSw ? null : _run,
              child: Text(_stage == _Stage.idle ? 'Run smoke test' : 'Run again'),
            ),
            const SizedBox(height: 12),
            if (_stage == _Stage.encodingHw || _stage == _Stage.encodingSw || _stage == _Stage.preparing)
              const Center(child: CircularProgressIndicator()),
            if (_stage == _Stage.done) ...[
              Text('PASSED — encoder: $_encoderUsed, time: ${_elapsed!.inMilliseconds} ms',
                  style: const TextStyle(color: Colors.green, fontWeight: FontWeight.bold)),
              Text('Output: $_outputPath', style: const TextStyle(fontSize: 11)),
              const SizedBox(height: 8),
              if (_player != null && _player!.value.isInitialized)
                AspectRatio(
                  aspectRatio: _player!.value.aspectRatio,
                  child: VideoPlayer(_player!),
                ),
            ],
            if (_stage == _Stage.failed)
              const Text('FAILED — see log below',
                  style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold)),
            const SizedBox(height: 12),
            Expanded(
              child: SingleChildScrollView(
                child: Text(_log, style: const TextStyle(fontFamily: 'monospace', fontSize: 11)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
