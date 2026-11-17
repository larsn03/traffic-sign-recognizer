import 'dart:math' as math;

/// The base model for a single extracted traffic-sign entry.
///
/// A [DataVector] bundles the semantic image features produced by the extractor
/// — colour counts, white-frame removal results, vertical/horizontal colour
/// change statistics, grid fractions and left/right asymmetry — together with
/// the information required to serialise them as one CSV record.
///
/// The numeric feature layout mirrors the CSV schema written by the Go
/// extractor (see `data_model/data_vector_v1.go`): a label and a path followed
/// by [featureCount] numeric columns, always in this order:
///
/// ```text
/// Number_of_different_colors, Contains_red, Contains_blue, contains_yellow,
/// contains_black, Number_of_color_changes_vertically_0..9,
/// Number_of_color_changes_horizontally_0..9, White_fraction_0..8,
/// Left_right_asymmetry_0..8
/// ```
class DataVector {  final String label;
  final String path;
  final int numberOfDifferentColors;
  final bool containsRed;
  final bool containsBlue;
  final bool containsYellow;
  final bool containsBlack;
  final List<int> numberOfColorChangesVertically;
  final List<int> numberOfColorChangesHorizontally;
  final List<double> whiteFractions;
  final List<double> leftRightAsymmetry;

  /// Number of numeric feature columns that follow the label and path.
  static const int featureCount =
      1 + // number of different colors
      4 + // contains_red / blue / yellow / black
      10 + // vertical color changes
      10 + // horizontal color changes
      9 + // white fractions
      9; // left/right asymmetry

  DataVector({
    required this.label,
    required this.path,
    this.numberOfDifferentColors = 0,
    this.containsRed = false,
    this.containsBlue = false,
    this.containsYellow = false,
    this.containsBlack = false,
    List<int>? numberOfColorChangesVertically,
    List<int>? numberOfColorChangesHorizontally,
    List<double>? whiteFractions,
    List<double>? leftRightAsymmetry,
  })  : numberOfColorChangesVertically = numberOfColorChangesVertically ?? const [],
        numberOfColorChangesHorizontally =
            numberOfColorChangesHorizontally ?? const [],
        whiteFractions = whiteFractions ?? const [],
        leftRightAsymmetry = leftRightAsymmetry ?? const [];

  /// Builds a [DataVector] from a label, path and a fully numeric feature list.
  ///
  /// The list is split back into the semantic fields using the documented
  /// column order so that [numericFeatures] reproduces the same sequence.
  factory DataVector.fromNumericFeatures(
    String label,
    String path,
    List<double> features,
  ) {
    if (features.length != featureCount) {
      throw ArgumentError('Expected $featureCount numeric features, got ${features.length}.');
    }

    return DataVector(
      label: label,
      path: path,
      numberOfDifferentColors: features[0].round(),
      containsRed: _asBool(features[1]),
      containsBlue: _asBool(features[2]),
      containsYellow: _asBool(features[3]),
      containsBlack: _asBool(features[4]),
      numberOfColorChangesVertically:
          features.sublist(5, 15).map((v) => v.round()).toList(),
      numberOfColorChangesHorizontally:
          features.sublist(15, 25).map((v) => v.round()).toList(),
      whiteFractions: features.sublist(25, 34),
      leftRightAsymmetry: features.sublist(34, featureCount),
    );
  }

  static bool _asBool(double v) => v != 0;

  /// The numeric features in CSV column order.
  List<double> get numericFeatures {
    final result = <double>[]
      ..add(numberOfDifferentColors.toDouble())
      ..add(containsRed ? 1 : 0)
      ..add(containsBlue ? 1 : 0)
      ..add(containsYellow ? 1 : 0)
      ..add(containsBlack ? 1 : 0);
    for (final v in numberOfColorChangesVertically) {
      result.add(v.toDouble());
    }
    for (final v in numberOfColorChangesHorizontally) {
      result.add(v.toDouble());
    }
    result.addAll(whiteFractions);
    result.addAll(leftRightAsymmetry);
    return result;
  }

  /// Whether this vector carries the full feature set (used to validate input).
  bool get isValidFeatures => numericFeatures.length == featureCount;

  /// A single CSV record (including label and path).
  List<String> toCsvRecord() {
    final numbers = numericFeatures.map(_formatNumber);
    return [label, path, ...numbers];
  }

  /// Parses one CSV record into a [DataVector].
  factory DataVector.fromCsvRecord(List<String> record) {
    if (record.length < 3) {
      throw ArgumentError('CSV record too short: ${record.length} columns.');
    }
    final features = record.sublist(2).map(double.parse).toList();
    return DataVector.fromNumericFeatures(record[0], record[1], features);
  }

  /// Builds a vector from a parsed CSV record (a list of string cells).
  ///
  /// This is the low-level helper used by [DataSet] when loading files. Unlike
  /// [fromCsvRecord] it does not assume the feature slice starts at index 2.
  factory DataVector.fromNumericRecord(List<String> record) {
    if (record.length < 2) {
      throw FormatException('CSV record too short to hold a label and path.');
    }
    final label = record[0];
    final path = record.length >= 3 ? record[1] : '';
    // Collapse any trailing cells past the expected width so callers that pass
    // a few extra columns still yield exactly [featureCount] features.
    var features = <double>[];
    for (final cell in record.sublist(2)) {
      if (cell.trim().isEmpty) continue;
      try {
        features.add(double.parse(cell));
      } on FormatException catch (e) {
        throw FormatException('Invalid numeric feature "${cell.trim()}": ${e.message}');
      }
    }
    while (features.length < featureCount) features.add(0.0);
    if (features.length > featureCount) features = features.sublist(0, featureCount);
    return DataVector.fromNumericFeatures(label, path, features);
  }
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is DataVector &&
          runtimeType == other.runtimeType &&
          label == other.label &&
          path == other.path &&
          numberOfDifferentColors == other.numberOfDifferentColors &&
          containsRed == other.containsRed &&
          containsBlue == other.containsBlue &&
          containsYellow == other.containsYellow &&
          containsBlack == other.containsBlack &&
          const ListEquality<int>().equals(
              numberOfColorChangesVertically,
              other.numberOfColorChangesVertically) &&
          const ListEquality<int>().equals(
              numberOfColorChangesHorizontally,
              other.numberOfColorChangesHorizontally) &&
          _doublesEqual(whiteFractions, other.whiteFractions) &&
          _doublesEqual(leftRightAsymmetry, other.leftRightAsymmetry));

  @override
  int get hashCode => Object.hash(
        label,
        path,
        numberOfDifferentColors,
        containsRed,
        containsBlue,
        containsYellow,
        containsBlack,
        _hash(numberOfColorChangesVertically),
        _hash(numberOfColorChangesHorizontally),
        _hash(whiteFractions.map((v) => v.hashCode).toList()),
        _hash(leftRightAsymmetry.map((v) => v.hashCode).toList()),
      );

  @override
  String toString() => 'DataVector{label: $label, path: $path}';
}

// --- small local value helpers so callers need not depend on package:collection

bool _doublesEqual(List<double> a, List<double> b) {
  if (a.length != b.length) return false;
  for (int i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

String _formatNumber(double value) {
  // Render whole numbers without a trailing `.0`, as the Go extractor renders
  // integers and short decimals.
  if (value == value.roundToDouble()) {
    return value.toInt().toString();
  }
  final rounded = (value * 1e6).round() / 1e6;
  return rounded.toString();
}

/// FNV-1a style hashing over an iterable, used only for [hashCode] so we do not
/// depend on an uncertain `Object.listHash` variant.
int _hash(Iterable<Object?> items) {
  const base = 0x811c9dc5;
  var h = base;
  final prime = 1677761;
  for (final it in items) {
    h ^= (it.hashCode & 0xffffffff) >>> 24 | (it.hashCode << 8) & 0xffffffff;
    h = (h ^ (it.hashCode % 256)) & 0xffffffff;
    h = (h * prime) & 0xffffffff;
  }
  return h;
}

/// Minimal local list equality/hash helper to avoid depending on
/// `package:collection` for value semantics.
class ListEquality<T> {
  const ListEquality();

  bool equals(List<T> a, List<T>? b) => listEquals(a, b);

  Object? listHash(Object? list) => (list ?? const []).hashCode;
}

bool listEquals<T>(List<T>? a, List<T>? b) {
  if (a == null || b == null) return identical(a, b);
  if (a.length != b.length) return false;
  for (int i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
