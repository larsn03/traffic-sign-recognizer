import 'dart:io';

import 'classification_result.dart';

/// An immutable collection of [ClassificationResult]s plus the aggregate metrics
/// derived from them.
///
/// It is what a [`ClassifierApi`](classifier_api.dart) implementation returns so
/// callers get both the per-entry verdicts and the summary accuracy in one shot.
/// Instances are value based: equality compares the result lists.
class ClassificationResultSet {
  /// The ordered per-entry results.
  final List<ClassificationResult> results;

  /// Iterator backing the underlying list convenience getters below.
  Iterator<ClassificationResult> get iterator => results.iterator;

  /// Total number of entries that were classified.
  int get total => results.length;

  /// Number of correctly classified entries.
  int get correctCount =>
      results.where((r) => r.correctlyClassified).length;

  /// Percentage (0–100) of correctly classified entries.
  double get accuracy => total == 0 ? 0.0 : correctCount * 100 / total;

  ClassificationResultSet(this.results);

  /// Convenience factory used by classifiers to wrap a plain list.
  factory ClassificationResultSet.fromList(List<ClassificationResult> results) {
    return ClassificationResultSet(results.toList());
  }

  /// Returns the faulty classifications (prediction differed from truth).
  Iterable<ClassificationResult> get faulty =>
      results.where((r) => !r.correctlyClassified);

  /// Total misclassifications.
  int get faultyCount => faulty.length;

  /// Renders each result to `Image <path> | Should be <x> | Classified as <y>`.
  List<String> toStringLines() => results.map((r) => r.stringify()).toList();

  /// Builds a plain set from any iterable of verdicts.
  factory ClassificationResultSet.fromResults(Iterable<ClassificationResult> src) {
    return ClassificationResultSet(src.toList());
  }

  /// Persists the results, one `stringify()` line per entry, to [path].
  void saveToFile(String path) {
    File(path).writeAsStringSync(toCsvString());
  }

  /// Renders a CSV document: header row plus one row per verdict with columns
  /// `path`, `shouldBe`, and `predicted`.
  String toCsvString() {
    final buffer = StringBuffer('path,shouldBe,predicted\n');
    for (final result in results) {
      buffer.writeln('${result.filePath},${result.shouldBe},${result.predicted}');
    }
    return buffer.toString();
  }
}
