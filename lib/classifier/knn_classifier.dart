import 'dart:math';

import '../data/data_set.dart';
import 'classifier_api.dart';
import 'classification_result.dart';
import 'classification_result_set.dart';

/// A k-nearest-neighbour classifier, backed by already classified data.
///
/// Before measuring distances every feature is standardised to zero mean and
/// unit variance using the statistics of the *training* set — this keeps large
/// magnitude colour-change counts from drowning out subtle signals such as the
/// asymmetry features when computing Euclidean distances.
class KnnClassifier implements ClassifierApi {
  /// The labelled vectors used to locate neighbours at query time.
  final DataSet alreadyClassifiedData;

  /// Number of neighbours consulted per query.
  final int k;

  KnnClassifier({DataSet? alreadyClassifiedData, this.k = 5})
      : alreadyClassifiedData = alreadyClassifiedData ?? DataSet();

  @override
  ClassificationResultSet classify(DataSet toBeClassified) {
    if (alreadyClassifiedData.isEmpty || toBeClassified.isEmpty) {
      return ClassificationResultSet(const []);
    }

    final stats = _Standardizer.compute(alreadyClassifiedData.data.map((v) => v.numericFeatures).toList());
    final neighbours = alreadyClassifiedData.length.clamp(1, k);

    final results = <ClassificationResult>[];
    for (final test in toBeClassified.data) {
      final prediction = _majorityLabel(
        stats.standardize(test.numericFeatures),
        alreadyClassifiedData.data.map((v) => v.numericFeatures).toList(),
        alreadyClassifiedData.data.map((v) => v.label).toList(),
        neighbours,
      );
      results.add(ClassificationResult(
        filePath: test.path,
        shouldBe: test.label,
        predicted: prediction,
      ));
    }

    return ClassificationResultSet.fromList(results);
  }

  /// Majority class among the [k] nearest standardised neighbours.
  String _majorityLabel(
    List<double> standardQuery,
    List<List<double>> trainFeatures,
    List<String> trainLabels,
    int n,
  ) {
    final scored = <int>[for (var i = 0; i < trainFeatures.length; i++) i];
    scored.sort((a, b) => _euclidean(standardQuery, trainFeatures[a]).compareTo(_euclidean(standardQuery, trainFeatures[b])));

    final votes = <String, int>{};
    for (var i = 0; i < n && i < scored.length; i++) {
      final label = trainLabels[scored[i]];
      votes[label] = (votes[label] ?? 0) + 1;
    }

    var bestLabel = '';
    var bestVotes = -1;
    for (final entry in votes.entries) {
      if (entry.value > bestVotes || (entry.value == bestVotes && entry.key.compareTo(bestLabel) < 0)) {
        bestVotes = entry.value;
        bestLabel = entry.key;
      }
    }
    return bestLabel;
  }

  /// Euclidean distance between two equal-length vectors.
  static double _euclidean(List<double> a, List<double> b) {
    var sum = 0.0;
    final length = a.length < b.length ? a.length : b.length;
    for (var i = 0; i < length; i++) {
      final diff = a[i] - b[i];
      sum += diff * diff;
    }
    return _sqrt(sum);
  }

  static double _sqrt(double value) => sqrt(value);
}

/// Per-feature mean and (population) standard deviation used to z-score a set of
/// numeric vectors. Columns with zero variance are left untransformed.
class _Standardizer {
  final List<double> mean;
  final List<double> stdDev;

  _Standardizer(this.mean, this.stdDev);

  static _Standardizer compute(List<List<double>> features) {
    if (features.isEmpty) {
      throw ArgumentError('Cannot standardise an empty feature set.');
    }
    final width = features.first.length;
    final mean = <double>[for (var i = 0; i < width; i++) 0.0];
    final stdDev = <double>[for (var i = 0; i < width; i++) 0.0];

    for (final row in features) {
      for (var i = 0; i < width && i < row.length; i++) {
        mean[i] += row[i];
      }
    }
    for (var i = 0; i < width; i++) {
      mean[i] /= features.length;
    }

    for (final row in features) {
      for (var i = 0; i < width && i < row.length; i++) {
        final diff = row[i] - mean[i];
        stdDev[i] += diff * diff;
      }
    }
    for (var i = 0; i < width; i++) {
      stdDev[i] = sqrt(stdDev[i] / features.length);
    }

    return _Standardizer(mean, stdDev);
  }

  /// Standardises a single vector using the stored statistics.
  List<double> standardize(List<double> row) {
    final result = <double>[for (var i = 0; i < row.length; i++) 0.0];
    for (var i = 0; i < row.length && i < mean.length; i++) {
      result[i] = stdDev[i] != 0 ? (row[i] - mean[i]) / stdDev[i] : row[i];
    }
    return result;
  }
}
