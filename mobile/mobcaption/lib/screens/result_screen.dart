import 'dart:io';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../services/gallery_export_service.dart';
import '../utils/file_paths.dart';
import 'home_screen.dart';

class ResultScreen extends StatefulWidget {
  final String outputPath;
  final JobPaths job;
  const ResultScreen({super.key, required this.outputPath, required this.job});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  late final VideoPlayerController _player;
  bool _ready = false;
  String? _message;

  @override
  void initState() {
    super.initState();
    _player = VideoPlayerController.file(File(widget.outputPath));
    _player.initialize().then((_) {
      if (!mounted) return;
      setState(() => _ready = true);
      _player.setLooping(true);
      _player.play();
    });
  }

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    try {
      await saveToGallery(widget.outputPath);
      if (mounted) setState(() => _message = 'Saved to gallery.');
    } catch (e) {
      if (mounted) setState(() => _message = 'Could not save: $e');
    }
  }

  Future<void> _share() async {
    await shareVideo(widget.outputPath);
  }

  Future<void> _startOver() async {
    await widget.job.cleanup();
    if (!mounted) return;
    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(builder: (_) => const HomeScreen()),
      (route) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Done')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            if (_ready)
              AspectRatio(aspectRatio: _player.value.aspectRatio, child: VideoPlayer(_player))
            else
              const AspectRatio(aspectRatio: 9 / 16, child: Center(child: CircularProgressIndicator())),
            const SizedBox(height: 16),
            if (_message != null)
              Padding(padding: const EdgeInsets.only(bottom: 12), child: Text(_message!)),
            Row(
              children: [
                Expanded(child: ElevatedButton(onPressed: _save, child: const Text('Save to gallery'))),
                const SizedBox(width: 8),
                Expanded(child: ElevatedButton(onPressed: _share, child: const Text('Share'))),
              ],
            ),
            const SizedBox(height: 8),
            TextButton(onPressed: _startOver, child: const Text('Start over')),
          ],
        ),
      ),
    );
  }
}
