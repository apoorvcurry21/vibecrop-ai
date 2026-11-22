import 'package:flutter/material.dart';

import 'screens/home_screen.dart';
import 'screens/settings_screen.dart';
import 'services/settings_store.dart';

void main() => runApp(const MobcaptionApp());

class MobcaptionApp extends StatelessWidget {
  const MobcaptionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'mobcaption',
      theme: ThemeData(colorSchemeSeed: Colors.deepPurple, useMaterial3: true),
      home: const StartupGate(),
    );
  }
}

/// First-run gate: if no Groq key is saved yet, Settings opens before Home.
class StartupGate extends StatefulWidget {
  const StartupGate({super.key});

  @override
  State<StartupGate> createState() => _StartupGateState();
}

class _StartupGateState extends State<StartupGate> {
  bool? _hasKey;

  @override
  void initState() {
    super.initState();
    SettingsStore().hasGroqApiKey().then((v) {
      if (mounted) setState(() => _hasKey = v);
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_hasKey == null) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return _hasKey! ? const HomeScreen() : const SettingsScreen(isFirstRun: true);
  }
}
