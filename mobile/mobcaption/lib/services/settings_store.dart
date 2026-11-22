import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Where the Groq API key (secret -> secure storage) and language code (not secret ->
/// shared_preferences) live on-device. No Gemini key: rewrite is out of scope for v1.
class SettingsStore {
  static const _groqKeyName = 'groq_api_key';
  static const _languageKeyName = 'caption_language';
  static const defaultLanguage = 'en';

  final FlutterSecureStorage _secure;

  SettingsStore({FlutterSecureStorage? secure}) : _secure = secure ?? const FlutterSecureStorage();

  Future<String?> getGroqApiKey() => _secure.read(key: _groqKeyName);

  Future<void> setGroqApiKey(String key) => _secure.write(key: _groqKeyName, value: key.trim());

  Future<bool> hasGroqApiKey() async {
    final key = await getGroqApiKey();
    return key != null && key.trim().isNotEmpty;
  }

  Future<String> getLanguage() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_languageKeyName) ?? defaultLanguage;
  }

  Future<void> setLanguage(String language) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_languageKeyName, language.trim().toLowerCase());
  }
}
