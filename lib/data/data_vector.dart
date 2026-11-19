/// The base model for a single data entry.
///
/// Mirrors the Go project's `DataVectorV1`: a fixed feature vector extracted
/// from one traffic-sign image. It is the single schema shared by extraction,
/// both classifiers, and the CSV exchange format.
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
  final int numberOfDifferentColors;

  final bool containsRed;
  final bool containsBlue;
  final bool containsYellow;
  final bool containsBlack;

  /// Color-change counts per vertical scan line (10 lines).
  final List<int> numberOfColorChangesVertically;

  /// Color-change counts per horizontal scan line (10 lines).
  final List<int> numberOfColorChangesHorizontally;

  /// White pixel fraction per 3x3 grid cell (9 cells, row-major).
  final List<double> whiteFractions;

  /// Left-right white asymmetry per 3x3 grid cell (9 cells, row-major).
  final List<double> leftRightAsymmetry;

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
  });

  /// The [featureCount] numeric features in canonical CSV order.
  ///
  /// The color-change counts are stored as `value / 10`, exactly as they
  /// appear in the CSV, so in-memory and on-disk representations agree.
  List<double> features() {
    return [
      numberOfDifferentColors.toDouble(),
      containsRed ? 1 : 0,
      containsBlue ? 1 : 0,
      containsYellow ? 1 : 0,
      containsBlack ? 1 : 0,
      for (final v in numberOfColorChangesVertically) v / 10,
      for (final v in numberOfColorChangesHorizontally) v / 10,
      ...whiteFractions,
      ...leftRightAsymmetry,
    ];
  }

  /// Encodes this vector as one CSV row (45 fields).
  List<String> toRow() {
    return [
      trafficSign,
      picturePath,
      numberOfDifferentColors.toString(),
      containsRed ? '1' : '0',
      containsBlue ? '1' : '0',
      containsYellow ? '1' : '0',
      containsBlack ? '1' : '0',
      for (final v in numberOfColorChangesVertically)
        (v / 10).toStringAsFixed(1),
      for (final v in numberOfColorChangesHorizontally)
        (v / 10).toStringAsFixed(1),
      for (final v in whiteFractions) v.toStringAsFixed(2),
      for (final v in leftRightAsymmetry) v.toStringAsFixed(2),
    ];
  }

  /// Decodes one CSV row (45 fields) into a [DataVector].
  factory DataVector.fromRow(List<String> fields) {
    bool flag(int i) => fields[i] == '1';
    return DataVector(
      picturePath: fields[1],
      trafficSign: fields[0],
      numberOfDifferentColors: int.parse(fields[2]),
      containsRed: flag(3),
      containsBlue: flag(4),
      containsYellow: flag(5),
      containsBlack: flag(6),
      numberOfColorChangesVertically: List<int>.generate(
        10,
        (i) => (double.parse(fields[7 + i]) * 10).round(),
      ),
      numberOfColorChangesHorizontally: List<int>.generate(
        10,
        (i) => (double.parse(fields[17 + i]) * 10).round(),
      ),
      whiteFractions:
          List<double>.generate(9, (i) => double.parse(fields[27 + i])),
      leftRightAsymmetry:
          List<double>.generate(9, (i) => double.parse(fields[36 + i])),
    );
  }
}
