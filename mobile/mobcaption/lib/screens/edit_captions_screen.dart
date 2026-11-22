import 'dart:io';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../captions/chunks.dart';
import '../models/caption_chunk.dart';
import '../services/video_probe.dart';
import '../utils/file_paths.dart';
import 'rendering_screen.dart';

/// CapCut/Insta-Edits-style caption editor, simplified to what was asked for: change a
/// caption's text, delete one, or add a new one at the current playhead. Not a draggable
/// timeline -- edits go through a plain list + a dialog for new captions.
class EditCaptionsScreen extends StatefulWidget {
  final JobPaths job;
  final String videoPath;
  final List<CaptionChunk> chunks;
  final VideoProbe probe;

  const EditCaptionsScreen({
    super.key,
    required this.job,
    required this.videoPath,
    required this.chunks,
    required this.probe,
  });

  @override
  State<EditCaptionsScreen> createState() => _EditCaptionsScreenState();
}

class _EditCaptionsScreenState extends State<EditCaptionsScreen> {
  late List<CaptionChunk> _chunks;
  late final VideoPlayerController _player;
  bool _playerReady = false;

  @override
  void initState() {
    super.initState();
    _chunks = List.of(widget.chunks);
    _player = VideoPlayerController.file(File(widget.videoPath));
    _player.initialize().then((_) {
      if (mounted) setState(() => _playerReady = true);
    });
  }

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  List<CaptionChunk> _sortedReindexed(List<CaptionChunk> chunks) {
    final sorted = List.of(chunks)..sort((a, b) => a.start.compareTo(b.start));
    return [for (var i = 0; i < sorted.length; i++) sorted[i].copyWith(id: i)];
  }

  void _editText(CaptionChunk chunk, String newText) {
    final text = newText.trim();
    if (text.isEmpty) return; // use the delete button to remove a caption, not an empty edit
    final words = retime(text, chunk.start, chunk.end, chunk.words);
    if (words.isEmpty) return;
    setState(() {
      final idx = _chunks.indexWhere((c) => c.id == chunk.id);
      if (idx == -1) return;
      _chunks[idx] = chunk.copyWith(words: words);
    });
  }

  void _delete(CaptionChunk chunk) {
    setState(() {
      _chunks.removeWhere((c) => c.id == chunk.id);
    });
  }

  Future<void> _addAtPlayhead() async {
    final pos = _playerReady ? _player.value.position.inMilliseconds / 1000.0 : 0.0;
    final result = await showDialog<_NewCaption>(
      context: context,
      builder: (_) => _AddCaptionDialog(initialStart: pos, initialEnd: pos + 1.5),
    );
    if (result == null) return;
    final words = retime(result.text, result.start, result.end, const []);
    if (words.isEmpty) return;
    setState(() {
      final newChunk = CaptionChunk(id: 0, start: result.start, end: result.end, words: words);
      _chunks = _sortedReindexed([..._chunks, newChunk]);
    });
  }

  void _goToRender() {
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => RenderingScreen(
        job: widget.job,
        videoPath: widget.videoPath,
        chunks: _sortedReindexed(_chunks),
        probe: widget.probe,
      ),
    ));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Edit captions')),
      floatingActionButton: FloatingActionButton(
        onPressed: _addAtPlayhead,
        tooltip: 'Add caption at playhead',
        child: const Icon(Icons.add),
      ),
      body: Column(
        children: [
          if (_playerReady)
            AspectRatio(
              aspectRatio: _player.value.aspectRatio,
              child: Stack(
                alignment: Alignment.bottomCenter,
                children: [
                  VideoPlayer(_player),
                  GestureDetector(
                    onTap: () => setState(() => _player.value.isPlaying ? _player.pause() : _player.play()),
                    child: Container(color: Colors.transparent),
                  ),
                  VideoProgressIndicator(_player, allowScrubbing: true),
                ],
              ),
            )
          else
            const AspectRatio(aspectRatio: 9 / 16, child: Center(child: CircularProgressIndicator())),
          const Divider(height: 1),
          Expanded(
            child: _chunks.isEmpty
                ? const Center(child: Text('No captions left. Tap + to add one.'))
                : ListView.builder(
                    itemCount: _chunks.length,
                    itemBuilder: (context, index) {
                      final chunk = _chunks[index];
                      final text = chunk.words.map((w) => w.text).join(' ');
                      return _CaptionRow(
                        key: ValueKey(chunk.id),
                        chunk: chunk,
                        initialText: text,
                        onChanged: (t) => _editText(chunk, t),
                        onDelete: () => _delete(chunk),
                      );
                    },
                  ),
          ),
          SafeArea(
            minimum: const EdgeInsets.all(12),
            child: ElevatedButton(
              onPressed: _chunks.isEmpty ? null : _goToRender,
              child: const Text('Burn in captions'),
            ),
          ),
        ],
      ),
    );
  }
}

class _CaptionRow extends StatefulWidget {
  final CaptionChunk chunk;
  final String initialText;
  final ValueChanged<String> onChanged;
  final VoidCallback onDelete;

  const _CaptionRow({
    super.key,
    required this.chunk,
    required this.initialText,
    required this.onChanged,
    required this.onDelete,
  });

  @override
  State<_CaptionRow> createState() => _CaptionRowState();
}

class _CaptionRowState extends State<_CaptionRow> {
  late final TextEditingController _controller = TextEditingController(text: widget.initialText);

  String _fmt(double s) {
    final d = Duration(milliseconds: (s * 1000).round());
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final sec = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$sec';
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(
            width: 44,
            child: Text(_fmt(widget.chunk.start), style: const TextStyle(fontSize: 11, color: Colors.grey)),
          ),
          Expanded(
            child: TextField(
              controller: _controller,
              decoration: const InputDecoration(isDense: true, border: OutlineInputBorder()),
              onSubmitted: widget.onChanged,
              onTapOutside: (_) => widget.onChanged(_controller.text),
            ),
          ),
          IconButton(icon: const Icon(Icons.delete_outline), onPressed: widget.onDelete),
        ],
      ),
    );
  }
}

class _NewCaption {
  final String text;
  final double start;
  final double end;
  const _NewCaption({required this.text, required this.start, required this.end});
}

class _AddCaptionDialog extends StatefulWidget {
  final double initialStart;
  final double initialEnd;
  const _AddCaptionDialog({required this.initialStart, required this.initialEnd});

  @override
  State<_AddCaptionDialog> createState() => _AddCaptionDialogState();
}

class _AddCaptionDialogState extends State<_AddCaptionDialog> {
  late final TextEditingController _text = TextEditingController();
  late final TextEditingController _start =
      TextEditingController(text: widget.initialStart.toStringAsFixed(2));
  late final TextEditingController _end = TextEditingController(text: widget.initialEnd.toStringAsFixed(2));

  @override
  void dispose() {
    _text.dispose();
    _start.dispose();
    _end.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Add caption'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          TextField(controller: _text, decoration: const InputDecoration(labelText: 'Text'), autofocus: true),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _start,
                  decoration: const InputDecoration(labelText: 'Start (s)'),
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: TextField(
                  controller: _end,
                  decoration: const InputDecoration(labelText: 'End (s)'),
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                ),
              ),
            ],
          ),
        ],
      ),
      actions: [
        TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Cancel')),
        TextButton(
          onPressed: () {
            final text = _text.text.trim();
            final start = double.tryParse(_start.text) ?? widget.initialStart;
            final end = double.tryParse(_end.text) ?? widget.initialEnd;
            if (text.isEmpty || end <= start) return;
            Navigator.of(context).pop(_NewCaption(text: text, start: start, end: end));
          },
          child: const Text('Add'),
        ),
      ],
    );
  }
}
