import 'package:test/test.dart';
import 'package:traffic_sign_recognizer/classifier/classification_result.dart';
import 'package:traffic_sign_recognizer/classifier/knn_classifier.dart';
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
  test('separates two well-separated clusters', () {
    final train = DataVectorSet(data: [
      for (var i = 0; i < 10; i++) vector('A', feature('A')),
      for (var i = 0; i < 10; i++) vector('B', feature('B')),
    ]);
    final test = DataVectorSet(data: [
      vector('A', feature('A')),
      vector('B', feature('B')),
    ]);
    final results = KnnClassifier(alreadyClassifiedData: train).classify(test);
    expect(results.accuracy, 100.0);
    expect(results.results[0].predicted, 'A');
    expect(results.results[1].predicted, 'B');
  });

  test('majority vote with alphabetical tie-break', () {
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
        KnnClassifier(alreadyClassifiedData: train, k: 2).classify(test);
    expect(results.results[0].predicted, 'a');
  });

  test('k falls back to 5 when <= 0', () {
    final zero = List<double>.filled(DataVector.featureCount, 0.0);
    final train = DataVectorSet(data: [
      for (var i = 0; i < 6; i++) vector('A', zero),
    ]);
    final test = DataVectorSet(data: [vector('A', zero)]);
    final results =
        KnnClassifier(alreadyClassifiedData: train, k: 0).classify(test);
    expect(results.accuracy, 100.0);
  });

  test('empty test set yields zero accuracy', () {
    final zero = List<double>.filled(DataVector.featureCount, 0.0);
    final train = DataVectorSet(data: [vector('A', zero)]);
    final results =
        KnnClassifier(alreadyClassifiedData: train).classify(DataVectorSet());
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
}

ClassificationResult resultsOf(String path, String shouldBe, String predicted) {
  return ClassificationResult(
      path: path, shouldBe: shouldBe, predicted: predicted);
}
