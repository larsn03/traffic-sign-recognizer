import 'dart:math' as math;

import '../data/data_vector.dart';
import '../data/data_vector_set.dart';
import 'classifier_api.dart';
import 'classification_result.dart';
import 'classification_result_set.dart';

/// A k-nearest-neighbour classifier, backed by already classified data.
///
/// Mirrors the Go project's `dataclassification` package: z-score
/// standardization fitted on the training set only (population standard
/// deviation, zero-variance columns mapped to 0), Euclidean distance,
/// majority vote with alphabetical tie-breaking.
class KnnClassifier implements ClassifierApi {
  DataVectorSet alreadyClassifiedData;

  /// Number of neighbours; values <= 0 fall back to the default of 5.
  final int k;

  KnnClassifier({DataVectorSet? alreadyClassifiedData, this.k = 5})
      : alreadyClassifiedData = alreadyClassifiedData ?? DataVectorSet();

  @override
  ClassificationResultSet classify(DataVectorSet toBeClassified) {
    final kk = k <= 0 ? 5 : k;
    final train = alreadyClassifiedData.data;
    final test = toBeClassified.data;
    if (test.isEmpty) {
      return ClassificationResultSet();
    }

    final numFeatures = DataVector.featureCount;
    final trainFeatures = [for (final v in train) v.features()];
    final testFeatures = [for (final v in test) v.features()];

    // Z-score standardization, fit on the training set only.
    final mean = List<double>.filled(numFeatures, 0);
    for (final f in trainFeatures) {
      for (var j = 0; j < numFeatures; j++) {
        mean[j] += f[j];
      }
    }
    for (var j = 0; j < numFeatures; j++) {
      mean[j] /= trainFeatures.length;
    }
    final variance = List<double>.filled(numFeatures, 0);
    for (final f in trainFeatures) {
      for (var j = 0; j < numFeatures; j++) {
        final d = f[j] - mean[j];
        variance[j] += d * d;
      }
    }
    final stddev = List<double>.generate(
        numFeatures, (j) => math.sqrt(variance[j] / trainFeatures.length));

    double transform(List<double> f, int j) =>
        stddev[j] == 0 ? 0 : (f[j] - mean[j]) / stddev[j];

    final trainStandardized = [
      for (final f in trainFeatures)
        [for (var j = 0; j < numFeatures; j++) transform(f, j)],
    ];
    final trainLabels = [for (final v in train) v.trafficSign];

    var correct = 0;
    final results = <ClassificationResult>[];
    for (var i = 0; i < test.length; i++) {
      final query = [
        for (var j = 0; j < numFeatures; j++) transform(testFeatures[i], j),
      ];
      final result = ClassificationResult(
        path: test[i].picturePath,
        shouldBe: test[i].trafficSign,
        predicted: _classifyOne(query, trainStandardized, trainLabels, kk),
      );
      if (result.isCorrect) correct++;
      results.add(result);
    }
    return ClassificationResultSet(
      results: results,
      accuracy: correct / test.length * 100,
    );
  }

  /// Euclidean distance to every training vector, majority vote over the k
  /// nearest; ties broken by alphabetical (lexicographically smallest) label.
  String _classifyOne(List<double> query, List<List<double>> train,
      List<String> labels, int k) {
    final distances = <double>[];
    for (var i = 0; i < train.length; i++) {
      var sum = 0.0;
      for (var j = 0; j < query.length; j++) {
        final d = query[j] - train[i][j];
        sum += d * d;
      }
      distances.add(math.sqrt(sum));
    }
    final order = List<int>.generate(distances.length, (i) => i)
      ..sort((a, b) => distances[a].compareTo(distances[b]));
    final votes = <String, int>{};
    for (var n = 0; n < k; n++) {
      final label = labels[order[n]];
      votes[label] = (votes[label] ?? 0) + 1;
    }
    var best = '';
    var bestVotes = -1;
    for (final label in votes.keys.toList()..sort()) {
      final count = votes[label]!;
      if (count > bestVotes) {
        bestVotes = count;
        best = label;
      }
    }
    return best;
  }
}
