import '../data/data_mask.dart';
import '../data/data_tree.dart';
import '../data/data_vector_set.dart';
import 'classifier_api.dart';
import 'classification_result.dart';
import 'classification_result_set.dart';
import 'learner_api.dart';

/// A classifier that uses a decision tree, which it can learn from data.
///
/// Features are scaled by the [DataMask] both when learning and when
/// predicting, so the tree's split thresholds (fit on masked features) and
/// the predicted-against features share the same scale.
class DecisionTreeClassifier implements ClassifierApi, LearnerApi<DataTree> {
  DataTree tree;

  /// Scale factor per feature family, applied to features before learning
  /// and prediction.
  final DataMask mask;

  DecisionTreeClassifier({DataTree? tree, DataMask? mask})
      : tree = tree ?? DataTree(),
        mask = mask ?? DataMask.identity();

  @override
  Future<ClassificationResultSet> classify(
      DataVectorSet toBeClassified) async {
    var correct = 0;
    final results = <ClassificationResult>[];
    for (final vector in toBeClassified.data) {
      final result = ClassificationResult(
        path: vector.picturePath,
        shouldBe: vector.trafficSign,
        predicted: tree.root.predict(mask.apply(vector.features())),
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
    tree = DataTree.train(dataSet, mask: mask);
    return tree;
  }
}
