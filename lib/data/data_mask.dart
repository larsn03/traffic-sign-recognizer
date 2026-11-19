import 'dart:io';

/// A per-family scale factor applied to data-vector features before
/// classification.
///
/// Each of the six feature families of a data vector is multiplied by its
/// factor. An all-`1.0` mask is the identity (no scaling). The mask is a
/// *post-extraction* transform: extraction writes raw features, and the mask
/// is applied only inside the classifiers.
///
/// The on-disk form is a small YAML file with one `family: factor` line per
/// family, e.g. `data-mask-example.yaml`.
class DataMask {
  /// The six feature families, in canonical feature order (matching
  /// `DataVector.features()`).
  static const List<String> familyNames = [
    'number_of_different_colors',
    'contains_colors',
    'number_of_changes_vertically',
    'number_of_changes_horizontally',
    'white_fraction',
    'left_right_asymmetry',
  ];

  /// Number of feature families (6).
  static const int familyCount = 6;

  /// Scale factor per family, aligned with [familyNames].
  final List<double> factors;

  DataMask(this.factors);

  /// A mask that leaves every feature unchanged.
  factory DataMask.identity() => DataMask(List.filled(familyCount, 1.0));

  /// Loads the mask from [path], creating and writing a default all-`1.0`
  /// mask when the file does not exist.
  static DataMask loadOrCreate(String path) {
    final file = File(path);
    if (!file.existsSync()) {
      final mask = DataMask.identity();
      mask.save(path);
      return mask;
    }
    return DataMask.parse(file.readAsStringSync());
  }

  /// Parses mask [content] (`family: factor` lines). Missing families default
  /// to `1.0`; unknown families are ignored; comments (`#`) are skipped.
  factory DataMask.parse(String content) {
    final factors = List<double>.filled(familyCount, 1.0);
    for (final rawLine in content.split('\n')) {
      var line = rawLine;
      final hash = line.indexOf('#');
      if (hash >= 0) line = line.substring(0, hash);
      line = line.trim();
      if (line.isEmpty) continue;
      final sep = line.indexOf(':');
      if (sep < 0) continue;
      final key = line.substring(0, sep).trim();
      final value = line.substring(sep + 1).trim();
      final index = familyNames.indexOf(key);
      if (index < 0) continue;
      final factor = double.tryParse(value);
      if (factor != null) factors[index] = factor;
    }
    return DataMask(factors);
  }

  /// Writes the mask to [path] as YAML, creating parent directories.
  void save(String path) {
    final buffer = StringBuffer()
      ..writeln('# Factors for each family of parameters');
    for (var i = 0; i < familyCount; i++) {
      buffer.writeln('${familyNames[i]}: ${factors[i]}');
    }
    final file = File(path);
    file.createSync(recursive: true);
    file.writeAsStringSync(buffer.toString(), mode: FileMode.write);
  }

  /// Multiplies each feature of [features] by its family's factor, returning
  /// a new list (the input is not modified).
  List<double> apply(List<double> features) {
    final out = List<double>.filled(features.length, 0);
    for (var j = 0; j < features.length; j++) {
      out[j] = features[j] * _factorFor(j);
    }
    return out;
  }

  /// The scale factor for feature index [j] (canonical feature order):
  /// `0` colors · `1..4` contains · `5..14` vertical · `15..24` horizontal ·
  /// `25..33` white fraction · `34..42` left-right asymmetry.
  double _factorFor(int j) {
    if (j == 0) return factors[0];
    if (j <= 4) return factors[1];
    if (j <= 14) return factors[2];
    if (j <= 24) return factors[3];
    if (j <= 33) return factors[4];
    return factors[5];
  }
}
