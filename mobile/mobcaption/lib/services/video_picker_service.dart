import 'dart:io';

import 'package:image_picker/image_picker.dart';

import '../utils/file_paths.dart';

/// Picks a video from the gallery and copies it into the job's own temp directory -- ffmpeg
/// needs a real filesystem path, and the picker's content:// URI isn't always directly usable
/// by native ffmpeg on Android.
Future<File?> pickVideoFromGallery(JobPaths job) async {
  final picked = await ImagePicker().pickVideo(source: ImageSource.gallery);
  if (picked == null) return null;
  final ext = picked.path.contains('.') ? picked.path.split('.').last : 'mp4';
  final dest = File(job.sourceVideo(ext));
  await dest.writeAsBytes(await picked.readAsBytes());
  return dest;
}
