import 'package:flutter_test/flutter_test.dart';

import 'package:mobcaption_smoke_test/main.dart';

void main() {
  testWidgets('shows the run button on launch', (WidgetTester tester) async {
    await tester.pumpWidget(const SmokeTestApp());
    expect(find.text('Run smoke test'), findsOneWidget);
  });
}
