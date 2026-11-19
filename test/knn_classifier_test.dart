import 'dart:math' as math;

import 'package:test/test.dart';
import 'package:traffic_sign_recognizer/classifier/classification_result.dart';
import 'package:traffic_sign_recognizer/classifier/knn_classifier.dart';
import 'package:traffic_sign_recognizer/data/data_mask.dart';
import 'package:traffic_sign_recognizer/data/data_vector.dart';
import 'package:traffic_sign_recognizer/data/data_vector_set.dart';

DataVector vector(String sign, List<double> features) {
  return DataVector(
    picturePath: '$sign.bmp',
    trafficSign: sign,
    numberOfDifferentColors: features[0].round(),
    containsRed: features[1] == 1,
    containsBlue: features[2] == 1,
    containsYellow: features[3] == 1,
    containsBlack: features[4] == 1,
    numberOfColorChangesVertically: List<int>.generate(
        10, (i) => (features[5 + i] * 10).round()),
    numberOfColorChangesHorizontally: List<int>.generate(
        10, (i) => (features[15 + i] * 10).round()),
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

  test('mask mode scales features (no z-score)', () async {
    // Cluster A at the origin; cluster B is far away in the color-count
    // family (feature 0) and slightly in the white-fraction family
    // (feature 25). The test point sits close to B in the color family.
    final zero = List<double>.filled(DataVector.featureCount, 0.0);
    final b = List<double>.filled(DataVector.featureCount, 0.0)
      ..[0] = 10.0
      ..[25] = 1.0;
    final query = List<double>.filled(DataVector.featureCount, 0.0)
      ..[0] = 9.0;
    final train = DataVectorSet(data: [
      for (var i = 0; i < 3; i++) vector('A', zero),
      for (var i = 0; i < 3; i++) vector('B', b),
    ]);
    final test = DataVectorSet(data: [vector('B', query)]);

    // Identity mask: the color family dominates -> nearest are the B's.
    final identity = await KnnClassifier(
      alreadyClassifiedData: train,
      k: 3,
      applyMask: true,
      applyZscore: false,
      mask: DataMask.identity(),
    ).classify(test);
    expect(identity.results[0].predicted, 'B');

    // Zeroing the color family flips the winner to A.
    final masked = await KnnClassifier(
      alreadyClassifiedData: train,
      k: 3,
      applyMask: true,
      applyZscore: false,
      mask: DataMask([0, 1, 1, 1, 1, 1]),
    ).classify(test);
    expect(masked.results[0].predicted, 'A');
  });

  test('mask-then-z-score mode', () async {
    // Same layout as the mask-mode test: A at the origin, B far away in the
    // color family (feature 0) and the white family (feature 25); the
    // query sits close to B in the color family.
    final zero = List<double>.filled(DataVector.featureCount, 0.0);
    final b = List<double>.filled(DataVector.featureCount, 0.0)
      ..[0] = 10.0
      ..[25] = 1.0;
    final query = List<double>.filled(DataVector.featureCount, 0.0)
      ..[0] = 9.0;
    final train = DataVectorSet(data: [
      for (var i = 0; i < 3; i++) vector('A', zero),
      for (var i = 0; i < 3; i++) vector('B', b),
    ]);
    final test = DataVectorSet(data: [vector('B', query)]);

    // With the identity mask, masking before z-scoring changes nothing.
    final z = await KnnClassifier(alreadyClassifiedData: train, k: 3)
        .classify(test);
    final zdmIdentity = await KnnClassifier(
      alreadyClassifiedData: train,
      k: 3,
      applyMask: true,
      applyZscore: true,
      mask: DataMask.identity(),
    ).classify(test);
    expect(z.results[0].predicted, 'B');
    expect(zdmIdentity.results[0].predicted, z.results[0].predicted);

    // Zeroing the color family *before* z-scoring flips the winner to A
    // (the color column becomes constant and maps to 0).
    final zdm = await KnnClassifier(
      alreadyClassifiedData: train,
      k: 3,
      applyMask: true,
      applyZscore: true,
      mask: DataMask([0, 1, 1, 1, 1, 1]),
    ).classify(test);
    expect(zdm.results[0].predicted, 'A');
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
