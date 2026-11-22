import 'dart:io';

import 'package:path_provider/path_provider.dart';

/// All intermediate files for one caption job live in their own temp subfolder, so retries
/// don't collide and "start over" cleanup is a single directory delete.
class JobPaths {
  final Directory dir;
  JobPaths._(this.dir);

  static Future<JobPaths> create() async {
    final tmp = await getTemporaryDirectory();
    final dir = Directory('${tmp.path}/mobcaption_${DateTime.now().millisecondsSinceEpoch}');
    await dir.create(recursive: true);
    return JobPaths._(dir);
  }

  String sourceVideo(String extension) => '${dir.path}/source.$extension';
  String get audio => '${dir.path}/audio.flac';
  String get overlayAss => '${dir.path}/overlay.ass';
  String get output => '${dir.path}/output.mp4';

  Future<void> cleanup() async {
    if (await dir.exists()) await dir.delete(recursive: true);
  }
}
