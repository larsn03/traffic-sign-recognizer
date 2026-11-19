import 'dart:io';
import 'dart:math';

import 'package:test/test.dart';
import 'package:traffic_sign_recognizer/classifier/decision_tree_classifier.dart';
import 'package:traffic_sign_recognizer/data/data_tree.dart';
import 'package:traffic_sign_recognizer/data/data_vector.dart';
import 'package:traffic_sign_recognizer/data/data_vector_set.dart';

/// Builds a random vector with a label derived from [features] so the
/// classes are learnable.
DataVector randomVector(List<double> features, String label) {
  return DataVector(
    picturePath: '$label-${Random().nextInt(1 << 30)}.bmp',
    trafficSign: label,
    numberOfDifferentColors: features[0],
    containsRed: features[1] > 0.5 ? 1 : 0,
    containsBlue: features[2] > 0.5 ? 1 : 0,
    containsYellow: features[3] > 0.5 ? 1 : 0,
    containsBlack: features[4] > 0.5 ? 1 : 0,
    numberOfColorChangesVertically:
        List<double>.generate(10, (i) => features[5 + i] * 10),
    numberOfColorChangesHorizontally:
        List<double>.generate(10, (i) => features[15 + i] * 10),
    whiteFractions: List<double>.generate(9, (i) => features[25 + i]),
    leftRightAsymmetry: List<double>.generate(9, (i) => features[34 + i]),
  );
}

void main() {
  test('smoke: training 5000 random samples does not crash', () {
    // Mirrors the Go project's TestTreeTrainRecover: a random synthetic
    // labelled dataset; the CART routine must run to completion.
    final random = Random(42);
    const labels = [
      'Stop',
      'Vorfahrt gewähren',
      'Vorfahrtsstraße',
      'Vorfahrt von rechts',
      'Fahrtrichtung links',
      'Fahrtrichtung rechts',
    ];
    final data = <DataVector>[];
    for (var i = 0; i < 5000; i++) {
      final label = labels[i % labels.length];
      // Bias one feature toward the label index so splits exist.
      final features = List<double>.generate(
          DataVector.featureCount, (_) => random.nextDouble());
      features[0] = labels.indexOf(label) / 5.0 + random.nextDouble() * 0.1;
      data.add(randomVector(features, label));
    }
    final tree = DataTree.train(DataVectorSet(data: data));
    expect(tree.root, isA<DataTreeNode>()); // a tree exists
    // Prediction must always return one of the training labels.
    for (final vector in data.sublist(0, 100)) {
      expect(labels, contains(tree.root.predict(vector.features)));
    }
  });

  test('learns a separable problem', () {
    final data = <DataVector>[];
    for (var i = 0; i < 40; i++) {
      data.add(randomVector(List<double>.filled(43, 0.0), 'A'));
      data.add(randomVector(List<double>.filled(43, 1.0), 'B'));
    }
    final tree = DataTree.train(DataVectorSet(data: data));
    expect(tree.root.predict(List<double>.filled(43, 0.0)), 'A');
    expect(tree.root.predict(List<double>.filled(43, 1.0)), 'B');
  });

  test('save/load round-trip preserves predictions', () {
    final data = <DataVector>[];
    for (var i = 0; i < 40; i++) {
      data.add(randomVector(List<double>.filled(43, 0.0), 'A'));
      data.add(randomVector(List<double>.filled(43, 1.0), 'B'));
    }
    final tree = DataTree.train(DataVectorSet(data: data));
    final path =
        '${Directory.systemTemp.createTempSync('tree-test').path}/tree.json';
    tree.save(path);
    final loaded = DataTree.load(path);
    final probe = List<double>.filled(43, 0.5);
    expect(loaded.root.predict(probe), tree.root.predict(probe));
    File(path).deleteSync();
  });

  test('save/load round-trip preserves standardization parameters', () {
    final data = <DataVector>[];
    for (var i = 0; i < 40; i++) {
      data.add(randomVector(List<double>.filled(43, 0.0), 'A'));
      data.add(randomVector(List<double>.filled(43, 1.0), 'B'));
    }
    final classifier = DecisionTreeClassifier()
      ..learnFromDataSet(DataVectorSet(data: data));
    final tree = classifier.tree;
    expect(tree.mean, isNotNull);
    expect(tree.stddev, isNotNull);

    final path =
        '${Directory.systemTemp.createTempSync('tree-test').path}/tree.json';
    tree.save(path);
    final loaded = DataTree.load(path);
    expect(loaded.mean, orderedEquals(tree.mean!));
    expect(loaded.stddev, orderedEquals(tree.stddev!));
    File(path).deleteSync();
  });

  test('majority breaks ties alphabetically', () {
    final leaf = DataTreeNode.leaf({'b': 2, 'a': 2, 'c': 1});
    expect(leaf.majority(), 'a');
    expect(leaf.predict(List<double>.filled(43, 0.0)), 'a');
  });

  test('classifier wraps tree prediction', () async {
    final data = <DataVector>[];
    for (var i = 0; i < 40; i++) {
      data.add(randomVector(List<double>.filled(43, 0.0), 'A'));
      data.add(randomVector(List<double>.filled(43, 1.0), 'B'));
    }
    final classifier = DecisionTreeClassifier()
      ..learnFromDataSet(DataVectorSet(data: data));
    final results = await classifier.classify(
      DataVectorSet(data: [
        randomVector(List<double>.filled(43, 0.0), 'A'),
        randomVector(List<double>.filled(43, 1.0), 'B'),
      ]),
    );
    expect(results.accuracy, 100.0);
  });
}
