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
    // The tree was trained on z-score standardized features; put the test
    // set on the same scale before routing it down the tree.
    final mean = tree.mean;
    final stddev = tree.stddev;
    if (mean != null && stddev != null) {
      toBeClassified.standardizeZ(mean: mean, stddev: stddev);
    }
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

  @override
  DataTree learnFromDataSet(DataVectorSet dataSet) {
    // Z-score standardize the training set in place and remember the
    // parameters with the tree, so test data is put on the same scale.
    final params = dataSet.standardizeZ();
    tree = DataTree.train(dataSet, mean: params.$1, stddev: params.$2);
    return tree;
  }
}
