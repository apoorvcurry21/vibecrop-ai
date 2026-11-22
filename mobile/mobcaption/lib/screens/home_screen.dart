import 'dart:io';

import 'package:flutter/material.dart';

import '../services/settings_store.dart';
import '../services/video_picker_service.dart';
import '../utils/file_paths.dart';
import 'settings_screen.dart';
import 'transcribing_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _video;
  JobPaths? _job;
  bool _picking = false;

  Future<void> _pick() async {
    setState(() => _picking = true);
    try {
      final job = await JobPaths.create();
      final video = await pickVideoFromGallery(job);
      if (video == null) {
        await job.cleanup();
        return;
      }
      await _job?.cleanup();
      if (!mounted) return;
      setState(() {
        _video = video;
        _job = job;
      });
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }

  Future<void> _generate() async {
    final store = SettingsStore();
    if (!await store.hasGroqApiKey()) {
      if (!mounted) return;
      await Navigator.of(context).push(MaterialPageRoute(builder: (_) => const SettingsScreen()));
      return;
    }
    if (_video == null || _job == null) return;
    if (!mounted) return;
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => TranscribingScreen(job: _job!, videoPath: _video!.path),
    ));
  }

  @override
  void dispose() {
    _job?.cleanup();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('mobcaption'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const SettingsScreen())),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton(
              onPressed: _picking ? null : _pick,
              child: Text(_video == null ? 'Pick a video' : 'Change video'),
            ),
            if (_video != null)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 12),
                child: Text(_video!.path.split(Platform.pathSeparator).last, textAlign: TextAlign.center),
              ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: _video == null ? null : _generate,
              child: const Text('Generate captions'),
            ),
          ],
        ),
      ),
    );
  }
}
