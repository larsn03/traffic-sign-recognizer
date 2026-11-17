import 'dart:convert';
import 'dart:io';

import '../classifier/classification_result.dart';
import '../classifier/classification_result_set.dart';
import '../data/data_set.dart';
import '../data/data_vector.dart';

/// A decision tree (CART) trained to predict a class label from numeric features.
///
/// [DataTree] is the concrete model used by
/// [`DecisionTreeClassifier`](../classifier/decision_tree_classifier.dart) both
/// as the learner result of [`LearnerApi`][learner] and as a classifier. It is
/// persisted to and loaded from JSON so it can travel between training and
/// classification phases, mirroring the Go extractor output.
///
/// Training uses the Gini impurity criterion with [maxDepth] and
/// [minSamplesToSplit] bounds identical in spirit to the Go implementation.
class DataTree {
  /// Maximum recursion depth used while growing the tree.
  static const int maxDepth = 12;

  /// Minimum number of samples required before a node may split.
  static const int minSamplesToSplit = 30;

  final _Node? _root;

  DataTree({_Node? root}) : _root = root;

  /// Whether the tree has been trained.
  bool get isTrained => _root != null;

  /// Builds a tree from an in-memory set of labelled examples.
  factory DataTree.fromData(List<_Sample> samples) {
    final root = _build(samples, depth: 0);
    return DataTree(root: root);
  }

  /// Convenience wrapper that trains directly on a [DataSet].
  static DataTree learn(DataSet dataSet) {
    final samples = dataSet.data.map(_sampleFromVector).toList();
    return DataTree.fromData(samples);
  }

  /// Predicts the class label for a single feature vector.
  String predict(List<double> features) {
    if (_root == null) {
      throw StateError('The tree is untrained.');
    }
    var node = _root;
    while (!node.isLeaf) {
      node = features[node.featureIndex!] <= node.threshold!
          ? node.left!
          : node.right!;
    }
    return node.label!;
  }

  /// Classifies every vector in [dataSet] using the trained tree.
  ClassificationResultSet classify(DataSet dataSet) {
    if (_root == null) {
      throw StateError('The tree must be trained before classification.');
    }
    final results = <ClassificationResult>[];
    for (final v in dataSet.data) {
      final prediction = predict(v.numericFeatures);
      results.add(
        ClassificationResult(
          filePath: v.path,
          shouldBe: v.label,
          predicted: prediction,
        ),
      );
    }
    return ClassificationResultSet(results);
  }

  /// Serialises the tree to JSON (root node first).
  String toJsonString() {
    final encoder = JsonEncoder.withIndent('  ');
    if (_root == null) return '[]';
    return encoder.convert([_root.toMap()]);
  }

  void saveToFile(String path) {
    final buffer = StringBuffer();
    buffer.write(toJsonString());
    // Simple write; callers may adapt to their IO layer.
    _writeFile(path, buffer.toString());
  }

  /// Loads a tree previously serialised with [toJsonString].
  static DataTree fromJsonString(String content) {
    final decoded = jsonDecode(content);
    if (decoded is List && decoded.isNotEmpty && decoded.first is Map) {
      return DataTree(root: _Node.fromMap(Map<String, dynamic>.from(decoded.first)));
    }
    throw FormatException('Unrecognised tree payload.');
  }

  static DataTree loadFromFile(String path) {
    final content = _readFile(path);
    if (content == null) throw StateError('Could not read tree file: $path');
    return fromJsonString(content);
  }
}

/// One labelled training example.
class _Sample {
  final List<double> features;
  final String label;

  _Sample(List<double> features, String label)
      : features = features,
        label = label;
}

_Sample _sampleFromVector(DataVector v) =>
    _Sample(v.numericFeatures, v.label);

/// Internal mutable node used only for tree construction/prediction.
class _Node {
  final int? featureIndex;
  final double? threshold;
  String? label; // leaf prediction
  Map<String, int>? counts; // class histogram at a leaf
  _Node? left;
  _Node? right;

  bool get isLeaf => label != null;

  _Node(this.featureIndex, this.threshold,
      {_Node? left, _Node? right})
      : left = left,
        right = right;

  _Node.leaf(String classLabel)
      : featureIndex = null,
        threshold = null,
        label = classLabel,
        counts = {classLabel: 1};

  Map<String, dynamic> toMap() => {
        if (featureIndex != null)
          'feature': featureIndex,
        if (threshold != null) 'threshold': threshold,
        if (label != null) 'label': label,
        if (counts != null) 'counts': counts,
        if (left != null) 'left': left!.toMap(),
        if (right != null) 'right': right!.toMap(),
      };

  factory _Node.fromMap(Map<String, dynamic> map) {
    final node = _Node(map['feature'] as int?, map['threshold'] as double?);
    node.label = map['label'] as String?;
    if (map.containsKey('counts')) {
      node.counts = (map['counts'] as Map).map(
        (k, v) => MapEntry(k.toString(), v as int),
      );
    }
    if (map.containsKey('left')) {
      node.left = _Node.fromMap(Map<String, dynamic>.from(map['left']));
    }
    if (map.containsKey('right')) {
      node.right = _Node.fromMap(Map<String, dynamic>.from(map['right']));
    }
    return node;
  }

  String majority() =>
      counts!.entries.reduce((a, b) =>
          b.value > a.value || (b.value == a.value && b.key.compareTo(a.key) < 0)
              ? b
              : a).key;
}

_Node _build(List<_Sample> samples, {required int depth}) {
  final labels = <String, int>{};
  for (final s in samples) {
    labels[s.label] = (labels[s.label] ?? 0) + 1;
  }

  if (depth >= DataTree.maxDepth ||
      samples.length < DataTree.minSamplesToSplit ||
      labels.length <= 1) {
    final leafCount = <String, int>{};
    for (final pair in labels.entries) {
      leafCount[pair.key] = pair.value;
    }
    return _Node.leaf(_majority(labels));
  }

  final split = _bestSplit(samples);
  final feature = split.first;
  final threshold = split.second;
  final improvement = split.third;
  if (improvement <= 0.0) {
    return _Node.leaf(_majority(labels));
  }

  final leftSamples = <_Sample>[];
  final rightSamples = <_Sample>[];
  for (final s in samples) {
    (s.features[feature] <= threshold ? leftSamples : rightSamples).add(s);
  }

  if (leftSamples.isEmpty || rightSamples.isEmpty) {
    return _Node.leaf(_majority(labels));
  }

  final node = _Node(feature, threshold);
  node.left = _build(leftSamples, depth: depth + 1);
  node.right = _build(rightSamples, depth: depth + 1);
  return node;
}

/// Majority class of a histogram (ties broken alphabetically).
String _majority(Map<String, int> counts) {
  var best = counts.keys.first;
  var bestCount = -1;
  for (final entry in counts.entries) {
    if (entry.value > bestCount ||
        (entry.value == bestCount && entry.key.compareTo(best) < 0)) {
      best = entry.key;
      bestCount = entry.value;
    }
  }
  return best;
}

/// Finds the single (feature, threshold) split that minimises weighted Gini.
Tuple3<int, double, double> _bestSplit(List<_Sample> samples) {
  const double nan = double.nan;
  final baseGini = _gini(
    <String, int>{},
    samples.map((s) => s.label),
  );

  var bestFeature = -1;
  var bestThreshold = nan;
  var bestImpurity = double.infinity;

  final featureCount = samples.first.features.length;
  for (var f = 0; f < featureCount; f++) {
    final values = samples.map((s) => s.features[f]).toList()
      ..sort();
    // A simple threshold sweep over midpoints between adjacent sorted values.
    var previous = double.infinity;
    for (final value in values) {
      if (value == previous) continue;
      final threshold = (value + previous) / 2.0;
      final (left, right) = _partition(samples, f, threshold);
      final impurity =
          _weightedGini(left) + _weightedGini(right);
      if (impurity < bestImpurity) {
        bestImpurity = impurity;
        bestFeature = f;
        bestThreshold = threshold;
      }
      previous = value;
    }
  }

  final improvement = baseGini - bestImpurity;
  return Tuple3(bestFeature, bestThreshold, improvement);
}

double _weightedGini(List<_Sample> samples) {
  final gini = _gini(
    <String, int>{},
    samples.map((s) => s.label),
  );
  return gini * samples.length; // weight already folded into the sweep
}

/// Computes Gini impurity from an iterable of labels.
double _gini(Map<String, int> histogram, Iterable<String> labels) {
  for (final label in labels) {
    histogram[label] = (histogram[label] ?? 0) + 1;
  }
  if (histogram.isEmpty) return 0.0;
  final total = histogram.values.fold<int>(0, (sum, c) => sum + c);
  var impurity = 1.0;
  for (final count in histogram.values) {
    final probability = count / total;
    impurity -= probability * probability;
  }
  return impurity;
}

/// Partitions [samples] by whether feature index [f] is at or below [threshold].
(List<_Sample>, List<_Sample>) _partition(
    List<_Sample> samples, int f, double threshold) {
  final left = <_Sample>[];
  final right = <_Sample>[];
  for (final s in samples) {
    if (s.features[f] <= threshold) {
      left.add(s);
    } else {
      right.add(s);
    }
  }
  return (left, right);
}

/// Tiny triple returned by [_bestSplit].
class Tuple3<A, B, C> {
  final A first;
  final B second;
  final C third;
  Tuple3(this.first, this.second, this.third);
}

/// Minimal file IO helper so [DataTree] stays transport-agnostic here.
String? _readFile(String path) => _SimpleIO.read(path);
void _writeFile(String path, String content) => _SimpleIO.write(path, content);

class _SimpleIO {
  static String read(String path) {
    return File(path).readAsStringSync();
  }

  static void write(String path, String content) {
    File(path).writeAsStringSync(content);
  }
}
