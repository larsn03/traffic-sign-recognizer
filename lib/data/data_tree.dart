import 'dart:convert';
import 'dart:io';

import 'data_vector_set.dart';

/// A decision tree trained with batch CART (Gini impurity).
///
/// Mirrors the Go project's `datatree` package: the tree is independent of
/// input row order (every split is evaluated over all samples at the node).
/// The tree may carry the z-score standardization parameters it was
/// trained with ([mean]/[stddev]), so new data can be put on the same scale
/// before prediction.
///
/// Serialization: pretty-printed JSON with the top-level keys `root`,
/// `mean`, `stddev` (the latter two omitted when absent). The node format
/// uses the exact keys `feature`, `thresh`, `left`, `right`, `counts`.
class DataTree {
  final DataTreeNode root;

  /// Z-score standardization parameters the tree was trained with, or null
  /// when the tree was trained on raw features.
  final List<double>? mean;
  final List<double>? stddev;

  DataTree({DataTreeNode? root, this.mean, this.stddev})
      : root = root ?? DataTreeNode.leaf(const <String, int>{});

  /// Trains a tree from [dataSet].
  ///
  /// [mean] and [stddev] are the standardization parameters the samples
  /// were standardized with; they are stored with the tree so new data can
  /// be put on the same scale before prediction.
  static DataTree train(
    DataVectorSet dataSet, {
    List<double>? mean,
    List<double>? stddev,
  }) {
    final samples = [
      for (final vector in dataSet.data)
        _Sample(vector.features, vector.trafficSign),
    ];
    final indices = List<int>.generate(samples.length, (i) => i);
    return DataTree(
      root: _build(samples, indices, 0),
      mean: mean,
      stddev: stddev,
    );
  }

  /// Writes the tree to [path] as pretty-printed JSON.
  void save(String path) {
    final encoded = const JsonEncoder.withIndent('  ').convert({
      'root': root.toJson(),
      if (mean != null) 'mean': mean,
      if (stddev != null) 'stddev': stddev,
    });
    File(path).writeAsStringSync(encoded);
  }

  /// Loads a tree previously written by [save]. Also accepts the legacy
  /// format in which the file contains the bare root node.
  static DataTree load(String path) {
    final json =
        jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;
    List<double>? standardization(String key) =>
        (json[key] as List<dynamic>?)
            ?.map((e) => (e as num).toDouble())
            .toList();
    return DataTree(
      root: DataTreeNode.fromJson((json['root'] ?? json) as Map<String, dynamic>),
      mean: standardization('mean'),
      stddev: standardization('stddev'),
    );
  }
}

/// Training parameters (per the project spec).
const int _minSamplesToSplit = 30;
const int _maxDepth = 12;

class _Sample {
  final List<double> features;
  final String label;
  _Sample(this.features, this.label);
}

class _Split {
  final int feature;
  final double thresh;
  final double gain;
  _Split(this.feature, this.thresh, this.gain);
}

DataTreeNode _build(
    List<_Sample> samples, List<int> indices, int depth) {
  final counts = <String, int>{};
  for (final i in indices) {
    final label = samples[i].label;
    counts[label] = (counts[label] ?? 0) + 1;
  }
  if (depth >= _maxDepth ||
      indices.length < _minSamplesToSplit ||
      counts.length <= 1) {
    return DataTreeNode.leaf(counts);
  }
  final split = _bestSplit(samples, indices, counts);
  if (split.feature < 0 || split.gain <= 0) {
    return DataTreeNode.leaf(counts);
  }
  final leftIndices = <int>[];
  final rightIndices = <int>[];
  for (final i in indices) {
    if (samples[i].features[split.feature] <= split.thresh) {
      leftIndices.add(i);
    } else {
      rightIndices.add(i);
    }
  }
  return DataTreeNode.internal(
    split.feature,
    split.thresh,
    _build(samples, leftIndices, depth + 1),
    _build(samples, rightIndices, depth + 1),
  );
}

/// Sweeps every feature for the split with the minimum weighted Gini.
///
/// For each feature the sample indices are sorted by that feature; candidate
/// split points lie between consecutive distinct values, with the threshold
/// set to the midpoint. Returns [feature] = -1 when no candidate exists.
_Split _bestSplit(
    List<_Sample> samples, List<int> indices, Map<String, int> totalCounts) {
  final n = indices.length;
  final base = _gini(counts: totalCounts, total: n);
  final numFeatures = samples[0].features.length;
  var bestFeature = -1;
  var bestThresh = 0.0;
  var bestWeighted = double.infinity;

  for (var feature = 0; feature < numFeatures; feature++) {
    final sorted = List<int>.of(indices)
      ..sort((a, b) {
        final c = samples[a].features[feature]
            .compareTo(samples[b].features[feature]);
        return c != 0 ? c : a.compareTo(b);
      });
    final leftCounts = <String, int>{};
    for (var i = 0; i < n - 1; i++) {
      final leftIndex = sorted[i];
      final label = samples[leftIndex].label;
      leftCounts[label] = (leftCounts[label] ?? 0) + 1;
      final vLeft = samples[leftIndex].features[feature];
      final vRight = samples[sorted[i + 1]].features[feature];
      if (vLeft == vRight) continue;
      var leftTotal = 0;
      for (final c in leftCounts.values) {
        leftTotal += c;
      }
      final rightTotal = n - leftTotal;
      final rightCounts = <String, int>{};
      for (final entry in totalCounts.entries) {
        final right = entry.value - (leftCounts[entry.key] ?? 0);
        if (right > 0) rightCounts[entry.key] = right;
      }
      final weighted = leftTotal / n *
              _gini(counts: leftCounts, total: leftTotal) +
          rightTotal / n * _gini(counts: rightCounts, total: rightTotal);
      if (weighted < bestWeighted) {
        bestWeighted = weighted;
        bestFeature = feature;
        bestThresh = (vLeft + vRight) / 2;
      }
    }
  }
  final gain = bestFeature < 0 ? 0.0 : base - bestWeighted;
  return _Split(bestFeature, bestThresh, gain);
}

/// Gini impurity: `1 - sum((count/total)^2)`; 0 when [total] is 0.
double _gini({required Map<String, int> counts, required int total}) {
  if (total == 0) return 0;
  var sum = 0.0;
  for (final count in counts.values) {
    final p = count / total;
    sum += p * p;
  }
  return 1 - sum;
}

/// A single node of a [DataTree].
///
/// Internal nodes have [feature] >= 0 and route samples left when
/// `features[feature] <= thresh`; leaves have [feature] == -1 and hold a
/// class histogram in [counts].
class DataTreeNode {
  final int feature;
  final double? thresh;
  final DataTreeNode? left;
  final DataTreeNode? right;
  final Map<String, int>? counts;

  DataTreeNode.internal(this.feature, this.thresh, this.left, this.right)
      : counts = null;

  DataTreeNode.leaf(this.counts)
      : feature = -1,
        thresh = null,
        left = null,
        right = null;

  /// The majority class; ties broken alphabetically (lexicographically
  /// smallest label wins), so results are deterministic.
  String majority() {
    final c = counts ?? const <String, int>{};
    String best = '';
    var bestCount = -1;
    for (final label in c.keys.toList()..sort()) {
      final count = c[label]!;
      if (count > bestCount) {
        bestCount = count;
        best = label;
      }
    }
    return best;
  }

  /// Routes [features] down to a leaf and returns its majority class.
  String predict(List<double> features) {
    var node = this;
    while (node.feature >= 0) {
      node = features[node.feature] <= node.thresh!
          ? node.left!
          : node.right!;
    }
    return node.majority();
  }

  Map<String, dynamic> toJson() {
    return {
      'feature': feature,
      if (thresh != null) 'thresh': thresh,
      if (left != null) 'left': left!.toJson(),
      if (right != null) 'right': right!.toJson(),
      if (counts != null) 'counts': counts,
    };
  }

  factory DataTreeNode.fromJson(Map<String, dynamic> json) {
    final feature = json['feature'] as int;
    final thresh = (json['thresh'] as num?)?.toDouble();
    final left = json['left'] == null
        ? null
        : DataTreeNode.fromJson(json['left'] as Map<String, dynamic>);
    final right = json['right'] == null
        ? null
        : DataTreeNode.fromJson(json['right'] as Map<String, dynamic>);
    final counts = json['counts'] == null
        ? null
        : (json['counts'] as Map<String, dynamic>)
            .map((k, v) => MapEntry(k, v as int));
    if (feature < 0) {
      return DataTreeNode.leaf(counts ?? const <String, int>{});
    }
    return DataTreeNode.internal(feature, thresh, left, right);
  }
}
