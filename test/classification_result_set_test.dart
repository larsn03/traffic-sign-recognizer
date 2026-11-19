import 'dart:math' as math;

import 'package:test/test.dart';
import 'package:traffic_sign_recognizer/classifier/classification_result.dart';
import 'package:traffic_sign_recognizer/classifier/classification_result_set.dart';

void main() {
  ClassificationResult result(String shouldBe, String predicted) =>
      ClassificationResult(path: 'p', shouldBe: shouldBe, predicted: predicted);

  ClassificationResultSet set(int correct, int faulty) =>
      ClassificationResultSet(results: [
        for (var i = 0; i < correct; i++) result('A', 'A'),
        for (var i = 0; i < faulty; i++) result('B', 'A'),
      ]);

  test('reports counts and the binomial proportion statistics', () {
    final results = set(4, 2);
    expect(results.total, 6);
    expect(results.correctCount, 4);

    final p = 2.0 / 3;
    final se = math.sqrt(p * (1 - p) / 6);
    expect(results.expectedValue, closeTo(p, 1e-12));
    expect(results.standardDeviation, closeTo(se, 1e-12));

    final ci = results.confidenceInterval;
    expect(ci.$1, closeTo(p - 1.96 * se, 1e-12));
    expect(ci.$2, 1.0); // p + 1.96*se > 1, clamped
  });

  test('an empty set has zero statistics', () {
    final results = ClassificationResultSet();
    expect(results.total, 0);
    expect(results.correctCount, 0);
    expect(results.expectedValue, 0);
    expect(results.standardDeviation, 0);
    expect(results.confidenceInterval, (0, 0));
  });

  test('a perfect set has a degenerate interval', () {
    final results = set(3, 0);
    expect(results.expectedValue, 1);
    expect(results.standardDeviation, 0);
    expect(results.confidenceInterval, (1, 1));
  });

  test('a fully faulty set has a degenerate interval', () {
    final results = set(0, 3);
    expect(results.expectedValue, 0);
    expect(results.standardDeviation, 0);
    expect(results.confidenceInterval, (0, 0));
  });
}
