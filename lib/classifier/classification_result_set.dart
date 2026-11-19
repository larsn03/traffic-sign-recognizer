import 'classification_result.dart';

/// A set of [ClassificationResult]s plus the overall accuracy.
class ClassificationResultSet {
  List<ClassificationResult> results;

  /// Accuracy in percent (correct / total * 100); 0 for an empty set.
  double accuracy;

  ClassificationResultSet({this.results = const [], this.accuracy = 0});

  void printResult() {
    for (final result in results) {
      print(result.stringify());
    }
  }
}
