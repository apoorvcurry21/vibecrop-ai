// Hormozi style's font (Impact) doesn't exist on Android -- Anton, a free bold-condensed
// Google Font, is bundled as an asset and extracted next to the .ass file at render time, so
// libass can find it via the subtitles filter's `fontsdir` option without a system install.

import 'dart:io';

import 'package:flutter/services.dart' show rootBundle;

const String _assetPath = 'assets/fonts/Anton-Regular.ttf';

/// Extracts the bundled fonts into `job.fontsDir`, creating it if needed. Safe to call every
/// render -- skips the copy if the file is already there from an earlier attempt on this job.
Future<String> ensureFontsDir(String fontsDirPath) async {
  final dir = Directory(fontsDirPath);
  if (!await dir.exists()) await dir.create(recursive: true);
  final dest = File('$fontsDirPath/Anton-Regular.ttf');
  if (!await dest.exists()) {
    final bytes = await rootBundle.load(_assetPath);
    await dest.writeAsBytes(bytes.buffer.asUint8List(bytes.offsetInBytes, bytes.lengthInBytes));
  }
  return fontsDirPath;
}
