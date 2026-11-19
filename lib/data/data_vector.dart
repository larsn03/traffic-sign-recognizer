import 'dart:math' as math;

/// The base model for a single data entry.
///
/// Mirrors the Go project's `DataVectorV1`: a fixed feature vector extracted
/// from one traffic-sign image. It is the single schema shared by extraction,
/// both classifiers, and the CSV exchange format.
///
/// Every numeric field is a [double]. The materialized [features] list is
/// the single source of truth for the numeric features: the standardization
/// methods on [DataVectorSet] rewrite it in place, while the raw typed
/// fields (and therefore the CSV encoding) stay untouched.
class DataVector {
  /// Number of numeric features per vector (after label + path):
  /// 1 color count + 4 flags + 10 vertical + 10 horizontal + 9 white
  /// fractions + 9 asymmetries.
  static const int featureCount = 43;

  /// Relative path of the image (e.g. `Stop/0/80x60/X0Y0.bmp`).
  final String picturePath;

  /// Class label (the name of the subfolder the image lives in).
  final String trafficSign;

  /// Number of distinct color categories present (red/blue/yellow/black/white).
  final double numberOfDifferentColors;

  final double containsRed;
  final double containsBlue;
  final double containsYellow;
  final double containsBlack;

  /// Color-change counts per vertical scan line (10 lines).
  final List<double> numberOfColorChangesVertically;

  /// Color-change counts per horizontal scan line (10 lines).
  final List<double> numberOfColorChangesHorizontally;

  /// White pixel fraction per 3x3 grid cell (9 cells, row-major).
  final List<double> whiteFractions;

  /// Left-right white asymmetry per 3x3 grid cell (9 cells, row-major).
  final List<double> leftRightAsymmetry;

  /// The [featureCount] numeric features in canonical order.
  ///
  /// Materialized once at construction. The standardization methods on
  /// [DataVectorSet] rewrite this list in place; the raw typed fields above
  /// are left untouched, so the CSV encoding ([toRow]) is unaffected by
  /// standardization.
  final List<double> features;

  DataVector({
    required this.picturePath,
    required this.trafficSign,
    required this.numberOfDifferentColors,
    required this.containsRed,
    required this.containsBlue,
    required this.containsYellow,
    required this.containsBlack,
    required this.numberOfColorChangesVertically,
    required this.numberOfColorChangesHorizontally,
    required this.whiteFractions,
    required this.leftRightAsymmetry,
  }) : features = [
          numberOfDifferentColors,
          containsRed,
          containsBlue,
          containsYellow,
          containsBlack,
          ...numberOfColorChangesVertically,
          ...numberOfColorChangesHorizontally,
          ...whiteFractions,
          ...leftRightAsymmetry,
        ];

  /// Rewrites [features] in place as `(x - mean) / stddev`; a column with
  /// zero standard deviation maps to 0.
  void standardize(List<double> mean, List<double> stddev) {
    for (var j = 0; j < features.length; j++) {
      final sd = stddev[j];
      features[j] = sd == 0 ? 0 : (features[j] - mean[j]) / sd;
    }
  }

  /// Rewrites [features] in place as `(x - min) / (max - min)`, mapping
  /// each column to [0, 1]; a column with zero range maps to 0.
  void minMaxStandardize(List<double> min, List<double> max) {
    for (var j = 0; j < features.length; j++) {
      final range = max[j] - min[j];
      features[j] = range == 0 ? 0 : (features[j] - min[j]) / range;
    }
  }

  /// Rewrites [features] in place as `x / ||x||_1` (L1 norm scaling), so
  /// the sum of the absolute values of the features is 1; a zero-norm
  /// vector stays all zeros.
  void l1Standardize() {
    var norm = 0.0;
    for (final f in features) {
      norm += f.abs();
    }
    if (norm == 0) return;
    for (var j = 0; j < features.length; j++) {
      features[j] /= norm;
    }
  }

  /// Rewrites [features] in place as `x / ||x||_2` (L2 norm scaling), so
  /// the sum of the squares of the features is 1; a zero-norm vector
  /// stays all zeros.
  void l2Standardize() {
    var norm = 0.0;
    for (final f in features) {
      norm += f * f;
    }
    norm = math.sqrt(norm);
    if (norm == 0) return;
    for (var j = 0; j < features.length; j++) {
      features[j] /= norm;
    }
  }

  /// Encodes this vector as one CSV row (45 fields).
  List<String> toRow() {
    return [
      trafficSign,
      picturePath,
      numberOfDifferentColors.round().toString(),
      containsRed.round() == 1 ? '1' : '0',
      containsBlue.round() == 1 ? '1' : '0',
      containsYellow.round() == 1 ? '1' : '0',
      containsBlack.round() == 1 ? '1' : '0',
      for (final v in numberOfColorChangesVertically) v.round().toString(),
      for (final v in numberOfColorChangesHorizontally)
        v.round().toString(),
      for (final v in whiteFractions) v.toStringAsFixed(2),
      for (final v in leftRightAsymmetry) v.toStringAsFixed(2),
    ];
  }

  /// Decodes one CSV row (45 fields) into a [DataVector].
  factory DataVector.fromRow(List<String> fields) {
    double flag(int i) => fields[i] == '1' ? 1 : 0;
    return DataVector(
      picturePath: fields[1],
      trafficSign: fields[0],
      numberOfDifferentColors: double.parse(fields[2]),
      containsRed: flag(3),
      containsBlue: flag(4),
      containsYellow: flag(5),
      containsBlack: flag(6),
      numberOfColorChangesVertically: List<double>.generate(
        10,
        (i) => double.parse(fields[7 + i]),
      ),
      numberOfColorChangesHorizontally: List<double>.generate(
        10,
        (i) => double.parse(fields[17 + i]),
      ),
      whiteFractions:
          List<double>.generate(9, (i) => double.parse(fields[27 + i])),
      leftRightAsymmetry:
          List<double>.generate(9, (i) => double.parse(fields[36 + i])),
    );
  }
}
