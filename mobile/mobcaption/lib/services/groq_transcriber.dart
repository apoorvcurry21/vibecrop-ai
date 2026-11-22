// Port of src/transcription/groq_whisper.py: calls Groq's hosted Whisper REST API directly
// from the phone (no backend). Same pre-flight checks and error cases as the Python original.

import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../models/transcript.dart';
import '../models/word.dart';
import '../utils/errors.dart';

const int maxAudioBytes = 25 * 1024 * 1024; // Groq free-tier upload limit
final RegExp _langRe = RegExp(r'^[a-z]{2,3}$');
const String _defaultModel = 'whisper-large-v3-turbo';
const String _endpoint = 'https://api.groq.com/openai/v1/audio/transcriptions';

Future<Transcript> transcribe({
  required File audioFile,
  required String apiKey,
  required String language,
  String model = _defaultModel,
}) async {
  final lang = language.trim().toLowerCase();
  if (!_langRe.hasMatch(lang)) {
    throw TranscriptionError("Invalid language code '$language'. Use an ISO code such as 'en' or 'hi'.");
  }
  if (!await audioFile.exists()) {
    throw TranscriptionError('Audio file not found: ${audioFile.path}');
  }
  final size = await audioFile.length();
  if (size > maxAudioBytes) {
    final mb = (size / 1e6).toStringAsFixed(1);
    throw TranscriptionError("Audio is $mb MB, over Groq's 25 MB limit. Use a shorter video.");
  }

  final request = http.MultipartRequest('POST', Uri.parse(_endpoint))
    ..headers['Authorization'] = 'Bearer $apiKey'
    ..fields['model'] = model
    ..fields['language'] = lang
    ..fields['response_format'] = 'verbose_json'
    ..fields['temperature'] = '0'
    ..files.add(await http.MultipartFile.fromPath('file', audioFile.path))
    // Two repeated fields of the same name (not a single comma-joined value) -- this is how
    // Groq's OpenAI-compatible endpoint expects timestamp_granularities as a multipart array.
    // http.MultipartRequest.fields is a Map (no duplicate keys), so these go through `files`
    // instead, via fromString parts with no filename, which IS a List and allows duplicates.
    ..files.add(http.MultipartFile.fromString('timestamp_granularities[]', 'word'))
    ..files.add(http.MultipartFile.fromString('timestamp_granularities[]', 'segment'));

  http.StreamedResponse streamed;
  try {
    streamed = await request.send();
  } on SocketException {
    throw const TranscriptionError('Could not reach Groq — check your internet connection.');
  } on HttpException {
    throw const TranscriptionError('Could not reach Groq — check your internet connection.');
  }

  final response = await http.Response.fromStream(streamed);

  if (response.statusCode == 429) {
    throw const TranscriptionError('Groq rate limit hit, wait a moment and try again.');
  }
  if (response.statusCode == 413) {
    throw const TranscriptionError('Audio too large for Groq, use a shorter clip.');
  }
  if (response.statusCode != 200) {
    final snippet = response.body.length > 300 ? response.body.substring(0, 300) : response.body;
    throw TranscriptionError('Groq transcription failed: ${response.statusCode} $snippet');
  }

  Map<String, dynamic> json;
  try {
    json = jsonDecode(response.body) as Map<String, dynamic>;
  } catch (e) {
    throw TranscriptionError('Groq returned an unreadable response: $e');
  }

  final transcript = _normalize(json, lang, model);
  if (transcript.words.isEmpty) {
    throw const TranscriptionError('Groq returned no words. Is there speech in the video?');
  }
  return transcript;
}

Transcript _normalize(Map<String, dynamic> response, String language, String model) {
  final words = [
    for (final w in (response['words'] as List? ?? []))
      TranscriptWord(
        word: (w['word'] ?? '').toString().trim(),
        start: _asDouble(w['start']),
        end: _asDouble(w['end']),
      ),
  ];
  final segments = [
    for (var i = 0; i < (response['segments'] as List? ?? []).length; i++)
      TranscriptSegment(
        id: (response['segments'][i]['id'] as num?)?.toInt() ?? i,
        start: _asDouble(response['segments'][i]['start']),
        end: _asDouble(response['segments'][i]['end']),
        text: (response['segments'][i]['text'] ?? '').toString().trim(),
      ),
  ];
  return Transcript(
    language: language,
    model: model,
    duration: _asDouble(response['duration']),
    text: (response['text'] ?? '').toString().trim(),
    segments: segments,
    words: words,
  );
}

double _asDouble(dynamic v) {
  if (v == null) return 0.0;
  if (v is num) return v.toDouble();
  return double.tryParse(v.toString()) ?? 0.0;
}
