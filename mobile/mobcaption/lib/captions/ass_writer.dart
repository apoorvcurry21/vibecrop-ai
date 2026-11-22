// Port of src/rendering/ass.py, trimmed to what v1 needs: no split-screen layout, no emphasis,
// no box/pop styles, no credits or text overlays — just the word-by-word highlighted caption
// line, exactly the "classic form" color markup the desktop app has always used.
//
// One real difference from the desktop app: it hardcodes PlayResX/PlayResY to 1080x1920 because
// its pipeline always normalizes footage into a vertical reel first. This app does no
// reframing — it burns captions onto the source video's own frame — so the caller passes the
// *actual* probed source resolution, and every size-like style field (tuned for a 1920-tall
// frame) is scaled by playResY/1920 so captions stay proportionally correct at any resolution.

import 'dart:io';
import 'dart:math' as math;

import '../models/caption_chunk.dart';
import 'styles.dart';

final RegExp _devanagari = RegExp(r'[ऀ-ॿ]');
const String _devaFont = 'Noto Sans Devanagari'; // Android has no "Nirmala UI" (that's Windows-only)
const int _tunedForHeight = 1920;
const int _tunedForWidth = 1080;

/// 'FF3B30' (RGB hex) -> the ASS colour '&H00303BFF&' (BGR).
String bgr(String rgb) {
  final r = rgb.toUpperCase();
  return '&H00${r.substring(4, 6)}${r.substring(2, 4)}${r.substring(0, 2)}&';
}

String _num(double x) {
  // Trim to 2 decimals so scaled outline/shadow values don't end up as long float reprs.
  final s = x.toStringAsFixed(2);
  return s.endsWith('.00') ? s.substring(0, s.length - 3) : s;
}

String _styleLine(String name, String font, int size, bool bold, String primary, String outlineColour,
    double outline, double shadow, int alignment, int marginL, int marginR, int marginV) {
  String c(String rgb) {
    final b = bgr(rgb);
    return b.substring(0, b.length - 1); // Style lines write colours without the closing '&'
  }

  return 'Style: $name,$font,$size,${c(primary)},${c(primary)},${c(outlineColour)},&H64000000,'
      '${bold ? -1 : 0},0,0,0,100,100,0,0,1,${_num(outline)},${_num(shadow)},$alignment,'
      '$marginL,$marginR,$marginV,1';
}

List<String> captionStyleLines(CaptionStyle st, {required int playResX, required int playResY}) {
  final heightScale = playResY / _tunedForHeight;
  final widthScale = playResX / _tunedForWidth;
  final marginLR = (60 * widthScale).round();
  final marginV = (st.marginV * heightScale).round();
  return [
    _styleLine('CaptionLatin', st.font, (st.size * heightScale).round(), st.bold, st.primary, '000000',
        st.outline * heightScale, st.shadow * heightScale, 2, marginLR, marginLR, marginV),
    _styleLine('CaptionDeva', _devaFont, (st.sizeDeva * heightScale).round(), true, st.primary, '000000',
        st.outline * heightScale, st.shadow * heightScale, 2, marginLR, marginLR, marginV),
  ];
}

String header(CaptionStyle st, {required int playResX, required int playResY}) {
  const fmt = 'Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, '
      'Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, '
      'Shadow, Alignment, MarginL, MarginR, MarginV, Encoding';
  final styleLines = captionStyleLines(st, playResX: playResX, playResY: playResY).join('\n');
  return '[Script Info]\n'
      'ScriptType: v4.00+\n'
      'PlayResX: $playResX\n'
      'PlayResY: $playResY\n'
      'WrapStyle: 0\n'
      'ScaledBorderAndShadow: yes\n'
      '\n'
      '[V4+ Styles]\n'
      'Format: $fmt\n'
      '$styleLines\n'
      '\n'
      '[Events]\n'
      'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n';
}

String assTime(double seconds) {
  final cs = (math.max(seconds, 0.0) * 100).round();
  final h = cs ~/ 360000;
  final remAfterH = cs % 360000;
  final m = remAfterH ~/ 6000;
  final remAfterM = remAfterH % 6000;
  final s = remAfterM ~/ 100;
  final c = remAfterM % 100;
  return '$h:${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}.${c.toString().padLeft(2, '0')}';
}

/// Strips characters libass would treat as override tags or line breaks.
String escape(String text) {
  final cleaned = text.replaceAll('{', '(').replaceAll('}', ')').replaceAll('\\', '/');
  return cleaned.split(RegExp(r'\s+')).where((t) => t.isNotEmpty).join(' ');
}

String _wordMarkup(String text, bool active, CaptionStyle st) {
  var body = escape(text);
  if (st.uppercase) body = body.toUpperCase();
  final color = active ? st.active : null; // classic.box is always false in v1, so "active" alone gates this
  if (color == null) return body;
  // The "classic form": open with the active colour, close back to the style's own primary colour.
  return '{\\c${bgr(color)}}$body{\\c${bgr(st.primary)}}';
}

/// One event per spoken word: the whole chunk's line is shown, with the currently-active word
/// recoloured. No split-screen slots and no emphasis overrides in v1, so (unlike the desktop
/// version) this never needs to cut a word's dialogue line at a layout boundary.
List<String> captionEvents(List<CaptionChunk> chunks, CaptionStyle st) {
  final events = <String>[];
  for (var ci = 0; ci < chunks.length; ci++) {
    final chunk = chunks[ci];
    final words = chunk.words;
    if (words.isEmpty) continue;
    final baseStyle = words.any((w) => _devanagari.hasMatch(w.text)) ? 'CaptionDeva' : 'CaptionLatin';
    final nextStart =
        (ci + 1 < chunks.length && chunks[ci + 1].words.isNotEmpty) ? chunks[ci + 1].words.first.start : null;

    for (var i = 0; i < words.length; i++) {
      final w = words[i];
      final start = w.start;
      var end = i + 1 < words.length ? words[i + 1].start : chunk.end + 0.15;
      if (i + 1 >= words.length && nextStart != null) {
        end = math.min(end, nextStart); // never overlap the next caption
      }
      end = math.max(end, start + 0.05);

      final line = [for (var j = 0; j < words.length; j++) _wordMarkup(words[j].text, j == i, st)].join(' ');
      events.add('Dialogue: 1,${assTime(start)},${assTime(end)},$baseStyle,,0,0,0,,$line');
    }
  }
  return events;
}

Future<File> writeAss(
  String path,
  List<String> events,
  CaptionStyle st, {
  required int playResX,
  required int playResY,
}) async {
  final file = File(path);
  final body = '${header(st, playResX: playResX, playResY: playResY)}${events.join('\n')}\n';
  await file.writeAsString(body);
  return file;
}
