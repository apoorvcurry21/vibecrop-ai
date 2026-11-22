import 'package:gal/gal.dart';
import 'package:share_plus/share_plus.dart';

Future<void> saveToGallery(String path) async {
  if (!await Gal.hasPermission()) {
    final granted = await Gal.requestPermission();
    if (!granted) return;
  }
  await Gal.putVideo(path, album: 'mobcaption');
}

Future<void> shareVideo(String path) async {
  await SharePlus.instance.share(ShareParams(files: [XFile(path)]));
}
