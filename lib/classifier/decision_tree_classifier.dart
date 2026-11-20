import '../data/data_tree.dart';
import '../data/data_vector_set.dart';
import 'classifier_api.dart';
import 'classification_result.dart';
import 'classification_result_set.dart';
import 'learner_api.dart';

/// A classifier that uses a decision tree, which it can learn from data.
class DecisionTreeClassifier implements ClassifierApi, LearnerApi<DataTree> {
  DataTree tree;

  DecisionTreeClassifier({DataTree? tree}) : tree = tree ?? DataTree();

  @override
  Future<ClassificationResultSet> classify(
      DataVectorSet toBeClassified) async {
    var correct = 0;
    final results = <ClassificationResult>[];
    for (final vector in toBeClassified.data) {
      final result = ClassificationResult(
        path: vector.picturePath,
        shouldBe: vector.trafficSign,
        predicted: tree.root.predict(vector.features),
      );
      if (result.isCorrect) correct++;
      results.add(result);
    }
    return ClassificationResultSet(
      results: results,
      accuracy: results.isEmpty ? 0 : correct / results.length * 100,
    );
  }

  /// Trains a tree from [dataSet]. [standardization] names the method the
  /// samples were standardized with (`'z'`, `'minmax'`, `'l1'`, `'l2'` or
  /// `'rs'`), and [mean]/[stddev] / [min]/[max] / [median]/[iqr] are the
  /// matching parameters (see [DataVectorSet.standardizeZ],
  /// [DataVectorSet.minMaxStandardize], [DataVectorSet.l1Standardize],
  /// [DataVectorSet.l2Standardize] and [DataVectorSet.robustStandardize]);
  /// they are stored with the tree so new data can be put on the same scale
  /// before prediction.
  @override
  DataTree learnFromDataSet(
    DataVectorSet dataSet, {
    String? standardization,
    List<double>? mean,
    List<double>? stddev,
    List<double>? min,
    List<double>? max,
    List<double>? median,
    List<double>? iqr,
  }) {
    tree = DataTree.train(dataSet,
        standardization: standardization,
        mean: mean,
        stddev: stddev,
        min: min,
        max: max,
        median: median,
        iqr: iqr);
    return tree;
  }
}
