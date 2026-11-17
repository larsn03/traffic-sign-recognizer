/// The result of classifying a single [DataVector].
///
/// A [ClassificationResult] records everything needed to audit a decision:
/// which source [path] it came from, what class it *should* have been (the true
/// label), and what the classifier predicted ([predicted]). Equality and hashing
/// are value based so results compare consistently regardless of order.
class ClassificationResult {
  /// Path or identifier of the entry that produced this result.
  final String filePath;

  /// The expected class label.
  final String shouldBe;

  /// The class the classifier assigned to the entry.
  final String predicted;

  /// True when [predicted] matches [shouldBe].
  bool get correctlyClassified => shouldBe == predicted;

  ClassificationResult({
    required this.filePath,
    required this.shouldBe,
    required this.predicted,
  });

  /// Canonical `Image <path> | Should be <x> | Classified as <y>` rendering.
  String stringify() =>
      'Image $filePath | Should be $shouldBe | Classified as $predicted';

  /// Like [stringify] but empty when the entry was classified correctly — handy
  /// for logging only the faulty decisions.
  String stringifyIfFaulty() =>
      correctlyClassified ? '' : stringify();

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is ClassificationResult &&
          runtimeType == other.runtimeType &&
          filePath == other.filePath &&
          shouldBe == other.shouldBe &&
          predicted == other.predicted);

  @override
  int get hashCode => Object.hash(filePath, shouldBe, predicted);

  @override
  String toString() => 'ClassificationResult($filePath)';
}
