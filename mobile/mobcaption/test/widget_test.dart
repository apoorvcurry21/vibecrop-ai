import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobcaption/main.dart';

void main() {
  testWidgets('app starts without throwing', (WidgetTester tester) async {
    await tester.pumpWidget(const MobcaptionApp());
    // Before settings_store resolves, the startup gate shows a spinner -- just confirm it builds.
    expect(find.byType(CircularProgressIndicator), findsOneWidget);
  });
}
