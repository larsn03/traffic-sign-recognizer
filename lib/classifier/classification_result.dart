/// The result of classifying a single data entry.
class ClassificationResult {
  /// Path of the classified image.
  final String path;

  /// The true label.
  final String shouldBe;

  /// The predicted label.
  final String predicted;

  ClassificationResult(
      {required this.path, required this.shouldBe, required this.predicted});

  bool get isCorrect => shouldBe == predicted;

  /// `Image <path> | Should be <label> | Classified as <label>`
  String stringify() =>
      'Image $path | Should be $shouldBe | Classified as $predicted';

  /// [stringify] only when misclassified, else the empty string.
  String stringifyIfFaulty() => isCorrect ? '' : stringify();
}
