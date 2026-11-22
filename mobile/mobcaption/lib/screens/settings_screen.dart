import 'package:flutter/material.dart';

import '../services/settings_store.dart';
import 'home_screen.dart';

class SettingsScreen extends StatefulWidget {
  final bool isFirstRun;
  const SettingsScreen({super.key, this.isFirstRun = false});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _store = SettingsStore();
  final _keyController = TextEditingController();
  final _langController = TextEditingController();
  bool _loading = true;
  bool _saving = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final key = await _store.getGroqApiKey();
    final lang = await _store.getLanguage();
    if (!mounted) return;
    setState(() {
      _keyController.text = key ?? '';
      _langController.text = lang;
      _loading = false;
    });
  }

  Future<void> _save() async {
    final key = _keyController.text.trim();
    final lang = _langController.text.trim().toLowerCase();
    if (key.isEmpty) {
      setState(() => _error = 'Enter your Groq API key.');
      return;
    }
    if (!RegExp(r'^[a-z]{2,3}$').hasMatch(lang)) {
      setState(() => _error = "Language must be a 2-3 letter ISO code, like 'en' or 'hi'.");
      return;
    }
    setState(() {
      _saving = true;
      _error = null;
    });
    await _store.setGroqApiKey(key);
    await _store.setLanguage(lang);
    if (!mounted) return;
    setState(() => _saving = false);
    if (widget.isFirstRun) {
      Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => const HomeScreen()));
    } else {
      Navigator.of(context).pop();
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('Groq API key', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 4),
            TextField(
              controller: _keyController,
              obscureText: true,
              decoration: const InputDecoration(border: OutlineInputBorder(), hintText: 'gsk_...'),
            ),
            const SizedBox(height: 16),
            const Text('Spoken language (ISO code)', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 4),
            TextField(
              controller: _langController,
              decoration: const InputDecoration(border: OutlineInputBorder(), hintText: 'en'),
            ),
            const Padding(
              padding: EdgeInsets.only(top: 4),
              child: Text(
                'Non-Latin scripts (e.g. Hindi) may render with a fallback font on some phones.',
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
            ),
            if (_error != null)
              Padding(
                padding: const EdgeInsets.only(top: 12),
                child: Text(_error!, style: const TextStyle(color: Colors.red)),
              ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _saving ? null : _save,
              child: _saving
                  ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Text('Save'),
            ),
          ],
        ),
      ),
    );
  }
}
