import 'package:flutter/material.dart';

import '../services/caption_pipeline.dart';
import '../services/settings_store.dart';
import '../utils/file_paths.dart';
import 'edit_captions_screen.dart';

class TranscribingScreen extends StatefulWidget {
  final JobPaths job;
  final String videoPath;
  const TranscribingScreen({super.key, required this.job, required this.videoPath});

  @override
  State<TranscribingScreen> createState() => _TranscribingScreenState();
}

class _TranscribingScreenState extends State<TranscribingScreen> {
  PipelineStage _stage = PipelineStage.extractingAudio;
  String? _error;

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    setState(() {
      _error = null;
      _stage = PipelineStage.extractingAudio;
    });
    try {
      final store = SettingsStore();
      final apiKey = await store.getGroqApiKey();
      final language = await store.getLanguage();
      final result = await transcribeVideo(
        job: widget.job,
        videoPath: widget.videoPath,
        apiKey: apiKey ?? '',
        language: language,
        onStage: (s) {
          if (mounted) setState(() => _stage = s);
        },
      );
      if (!mounted) return;
      Navigator.of(context).pushReplacement(MaterialPageRoute(
        builder: (_) => EditCaptionsScreen(
          job: widget.job,
          videoPath: widget.videoPath,
          chunks: result.chunks,
          probe: result.probe,
        ),
      ));
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  String _label(PipelineStage s) {
    switch (s) {
      case PipelineStage.extractingAudio:
        return 'Extracting audio...';
      case PipelineStage.transcribing:
        return 'Transcribing (Groq)...';
      case PipelineStage.buildingCaptions:
        return 'Building captions...';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Generating captions')),
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
                  const CircularProgressIndicator(),
                  const SizedBox(height: 16),
                  Text(_label(_stage)),
                ],
              ),
      ),
    );
  }
}
