import 'dart:io';
import 'dart:math' as math;

import 'package:csv/csv.dart';

import 'data_vector.dart';

/// A collection of [DataVector]s that can be classified.
///
/// Mirrors the Go project's `DataSet`: the single schema shared by
/// extraction, KNN and the decision tree, exchanged on disk as CSV.
class DataVectorSet {
  List<DataVector> data;

  DataVectorSet({this.data = const []});

  /// The canonical CSV header (45 column names).
  static final List<String> header = [
    'Traffic_Sign',
    'Path',
    'Number_of_different_colors',
    'Contains_red',
    'Contains_blue',
    'Contains_yellow',
    'Contains_black',
    for (var i = 0; i < 10; i++) 'Number_of_color_changes_vertically_$i',
    for (var i = 0; i < 10; i++) 'Number_of_color_changes_horizontally_$i',
    for (var i = 0; i < 9; i++) 'White_fraction_$i',
    for (var i = 0; i < 9; i++) 'Left_right_asymmetry_$i',
  ];

  /// Z-score standardizes every vector in this set, in place.
  ///
  /// Each of the [DataVector.featureCount] feature columns is transformed
  /// to `(x - mean) / stddev`, where [mean] and [stddev] are the column
  /// mean and population standard deviation. When they are not supplied
  /// they are fit on this set; supply parameters fitted elsewhere (e.g. on
  /// a training set) to apply them to this set instead. Zero-variance
  /// columns map to 0.
  ///
  /// Only the in-memory [DataVector.features] values are rewritten — the
  /// raw typed fields, and therefore the CSV encoding, are untouched.
  /// Returns the (mean, stddev) actually applied.
  (List<double>, List<double>) standardizeZ({
    List<double>? mean,
    List<double>? stddev,
  }) {
    final n = DataVector.featureCount;
    if (mean == null || stddev == null) {
      final m = List<double>.filled(n, 0);
      final v = List<double>.filled(n, 0);
      if (data.isNotEmpty) {
        for (final vector in data) {
          final f = vector.features;
          for (var j = 0; j < n; j++) {
            m[j] += f[j];
          }
        }
        for (var j = 0; j < n; j++) {
          m[j] /= data.length;
        }
        for (final vector in data) {
          final f = vector.features;
          for (var j = 0; j < n; j++) {
            final d = f[j] - m[j];
            v[j] += d * d;
          }
        }
        for (var j = 0; j < n; j++) {
          v[j] = math.sqrt(v[j] / data.length);
        }
      }
      mean = m;
      stddev = v;
    }
    for (final vector in data) {
      vector.standardize(mean, stddev);
    }
    return (mean, stddev);
  }

  /// Writes the set to [pathToCsv] with the canonical header row, one row
  /// per successfully extracted vector. Uses LF line endings (matching the
  /// Go reference project's `encoding/csv` default and the multi-process
  /// merge path).
  void toCsv(String pathToCsv) {
    final rows = <List<dynamic>>[
      header,
      for (final vector in data) vector.toRow(),
    ];
    final encoder = CsvEncoder(lineDelimiter: '\n');
    File(pathToCsv).writeAsStringSync(encoder.convert(rows));
  }

  /// Reads a set from [pathToCsv]; the header row is skipped, as are rows
  /// with fewer than 3 fields.
  static DataVectorSet fromCsv(String pathToCsv) {
    final rows = csv.decode(File(pathToCsv).readAsStringSync());
    final data = <DataVector>[];
    for (final row in rows.skip(1)) {
      if (row.length < 3) continue;
      data.add(DataVector.fromRow(row.cast<String>()));
    }
    return DataVectorSet(data: data);
  }
}
