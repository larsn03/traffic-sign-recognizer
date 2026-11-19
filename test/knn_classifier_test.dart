import 'dart:math' as math;

import 'package:test/test.dart';
import 'package:traffic_sign_recognizer/classifier/classification_result.dart';
import 'package:traffic_sign_recognizer/classifier/knn_classifier.dart';
import 'package:traffic_sign_recognizer/data/data_vector.dart';
import 'package:traffic_sign_recognizer/data/data_vector_set.dart';

DataVector vector(String sign, List<double> features) {
  return DataVector(
    picturePath: '$sign.bmp',
    trafficSign: sign,
    numberOfDifferentColors: features[0],
    containsRed: features[1],
    containsBlue: features[2],
    containsYellow: features[3],
    containsBlack: features[4],
    numberOfColorChangesVertically:
        List<double>.generate(10, (i) => features[5 + i]),
    numberOfColorChangesHorizontally:
        List<double>.generate(10, (i) => features[15 + i]),
    whiteFractions: List<double>.generate(9, (i) => features[25 + i]),
    leftRightAsymmetry: List<double>.generate(9, (i) => features[34 + i]),
  );
}

List<double> feature(String sign) {
  final f = List<double>.filled(DataVector.featureCount, 0);
  if (sign == 'A') {
    // Cluster A sits at the origin.
  } else {
    // Cluster B is far away in every dimension.
    for (var i = 0; i < f.length; i++) {
      f[i] = 1.0;
    }
  }
  return f;
}

void main() {
  test('separates two well-separated clusters', () async {
    final train = DataVectorSet(data: [
      for (var i = 0; i < 10; i++) vector('A', feature('A')),
      for (var i = 0; i < 10; i++) vector('B', feature('B')),
    ]);
    final test = DataVectorSet(data: [
      vector('A', feature('A')),
      vector('B', feature('B')),
    ]);
    final results =
        await KnnClassifier(alreadyClassifiedData: train).classify(test);
    expect(results.accuracy, 100.0);
    expect(results.results[0].predicted, 'A');
    expect(results.results[1].predicted, 'B');
  });

  test('majority vote with alphabetical tie-break', () async {
    // Query is equidistant from one 'b' and one 'a' sample; with k=2 the
    // vote is 1:1 and the lexicographically smallest label must win.
    final zero = List<double>.filled(DataVector.featureCount, 0.0);
    final one = List<double>.filled(DataVector.featureCount, 0.0);
    one[0] = 2.0;
    final half = List<double>.filled(DataVector.featureCount, 0.0);
    half[0] = 1.0;
    final train = DataVectorSet(data: [
      vector('b', zero),
      vector('a', one),
    ]);
    final test = DataVectorSet(data: [vector('a', half)]);
    final results =
        await KnnClassifier(alreadyClassifiedData: train, k: 2).classify(test);
    expect(results.results[0].predicted, 'a');
  });

  test('k falls back to 5 when <= 0', () async {
    final zero = List<double>.filled(DataVector.featureCount, 0.0);
    final train = DataVectorSet(data: [
      for (var i = 0; i < 6; i++) vector('A', zero),
    ]);
    final test = DataVectorSet(data: [vector('A', zero)]);
    final results =
        await KnnClassifier(alreadyClassifiedData: train, k: 0).classify(test);
    expect(results.accuracy, 100.0);
  });

  test('empty test set yields zero accuracy', () async {
    final zero = List<double>.filled(DataVector.featureCount, 0.0);
    final train = DataVectorSet(data: [vector('A', zero)]);
    final results = await KnnClassifier(alreadyClassifiedData: train)
        .classify(DataVectorSet());
    expect(results.accuracy, 0.0);
    expect(results.results, isEmpty);
  });

  test('result string format', () {
    final r = resultsOf('Stop/0/80x60/X0Y0.bmp', 'Stop', 'Stop');
    expect(r.stringify(),
        'Image Stop/0/80x60/X0Y0.bmp | Should be Stop | Classified as Stop');
    expect(r.stringifyIfFaulty(), '');
    final faulty = resultsOf('p.bmp', 'Stop', 'Vorfahrt von rechts');
    expect(faulty.stringifyIfFaulty(),
        'Image p.bmp | Should be Stop | Classified as Vorfahrt von rechts');
  });

  test('parallel classification matches the sequential run', () async {
    // Deterministic pseudo-random dataset: two overlapping clusters, so
    // predictions are non-trivial and tie-breaks can occur.
    final rng = math.Random(42);
    DataVector randomVector(String sign) {
      final f = List<double>.generate(
          DataVector.featureCount, (j) => rng.nextDouble() * 2 - 1);
      // Push the cluster apart in a few dimensions.
      f[0] = sign == 'A' ? f[0] * 0.2 : 1 + f[0] * 0.2;
      f[1] = sign == 'B' ? f[1] * 0.2 : -1 + f[1] * 0.2;
      return vector(sign, f);
    }

    final train = DataVectorSet(data: [
      for (var i = 0; i < 300; i++) randomVector(i.isEven ? 'A' : 'B'),
    ]);
    final test = DataVectorSet(data: [
      for (var i = 0; i < 1200; i++) randomVector(i % 3 == 0 ? 'B' : 'A'),
    ]);

    final sequential = await KnnClassifier(
      alreadyClassifiedData: train,
      minRowsForIsolates: 1 << 30, // force the sequential path
    ).classify(test);
    final parallel = await KnnClassifier(
      alreadyClassifiedData: train,
      minRowsForIsolates: 1, // force the isolate pool path
    ).classify(test);

    expect(parallel.results.length, test.data.length);
    expect(parallel.accuracy, sequential.accuracy);
    for (var i = 0; i < test.data.length; i++) {
      expect(parallel.results[i].path, sequential.results[i].path);
      expect(parallel.results[i].shouldBe, sequential.results[i].shouldBe);
      expect(parallel.results[i].predicted, sequential.results[i].predicted,
          reason: 'row $i');
    }
  });
}

ClassificationResult resultsOf(String path, String shouldBe, String predicted) {
  return ClassificationResult(
      path: path, shouldBe: shouldBe, predicted: predicted);
}
