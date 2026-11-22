import 'package:flutter/material.dart';

import '../captions/styles.dart';
import '../models/caption_chunk.dart';
import '../services/caption_pipeline.dart';
import '../services/video_probe.dart';
import '../utils/file_paths.dart';
import 'result_screen.dart';

class RenderingScreen extends StatefulWidget {
  final JobPaths job;
  final String videoPath;
  final List<CaptionChunk> chunks;
  final VideoProbe probe;
  final CaptionStyle style;

  const RenderingScreen({
    super.key,
    required this.job,
    required this.videoPath,
    required this.chunks,
    required this.probe,
    required this.style,
  });

  @override
  State<RenderingScreen> createState() => _RenderingScreenState();
}

class _RenderingScreenState extends State<RenderingScreen> {
  double _progress = 0;
  String? _error;

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    setState(() {
      _error = null;
      _progress = 0;
    });
    try {
      final result = await renderVideo(
        job: widget.job,
        videoPath: widget.videoPath,
        chunks: widget.chunks,
        probe: widget.probe,
        style: widget.style,
        onProgress: (f) {
          if (mounted) setState(() => _progress = f);
        },
      );
      if (!mounted) return;
      Navigator.of(context).pushReplacement(MaterialPageRoute(
        builder: (_) => ResultScreen(outputPath: result.outputPath, job: widget.job),
      ));
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Burning in captions')),
      body: Center(
        child: _error != null
            ? Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(_error!, style: const TextStyle(color: Colors.red), textAlign: TextAlign.center),
                    const SizedBox(height: 16),
                    ElevatedButton(onPressed: _run, child: const Text('Try again')),
                  ],
                ),
              )
            : Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(value: _progress > 0 ? _progress : null),
                  const SizedBox(height: 16),
                  Text('${(_progress * 100).toStringAsFixed(0)}%'),
                ],
              ),
      ),
    );
  }
}
