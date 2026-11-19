import 'dart:math' as math;

import 'classification_result.dart';

/// A set of [ClassificationResult]s plus the overall accuracy.
class ClassificationResultSet {
  List<ClassificationResult> results;

  /// Accuracy in percent (correct / total * 100); 0 for an empty set.
  double accuracy;

  ClassificationResultSet({this.results = const [], this.accuracy = 0});

  /// The number of correctly classified results.
  int get correctCount => results.where((r) => r.isCorrect).length;

  /// The total number of results.
  int get total => results.length;

  /// The expected value of the accuracy: the sample proportion of correct
  /// results, in [0, 1] — the estimator of the true accuracy. 0 for an
  /// empty set.
  double get expectedValue => total == 0 ? 0 : correctCount / total;

  /// The standard deviation (standard error) of the accuracy proportion:
  /// `sqrt(p * (1 - p) / n)`. 0 for an empty set.
  double get standardDeviation {
    if (total == 0) return 0;
    final p = expectedValue;
    return math.sqrt(p * (1 - p) / total);
  }

  /// The 95% confidence interval of the accuracy proportion:
  /// `p +/- 1.96 * standardDeviation`, clamped to [0, 1]. (0, 0) for an
  /// empty set.
  (double, double) get confidenceInterval {
    if (total == 0) return (0, 0);
    final p = expectedValue;
    final margin = 1.96 * standardDeviation;
    return (
      (p - margin).clamp(0.0, 1.0),
      (p + margin).clamp(0.0, 1.0),
    );
  }
}
