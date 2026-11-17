import 'dart:io';

import '../data/data_set.dart';
import '../data/data_tree.dart';
import 'classifier_api.dart';
import 'classification_result_set.dart';
import 'learner_api.dart';

/// A classifier that uses a decision tree, which it can learn from data.
///
/// It satisfies both [`ClassifierApi`](classifier_api.dart) — by delegating to
/// the trained [DataTree] when [classify] is called — and
/// [`LearnerApi<DataTree>`][], which returns the learned model. Training simply
/// wraps a [`DataTree.learn`][learn]; classification mirrors the Go CLI by
/// optionally writing every verdict (its own `Image … | Should be … | Classified
/// as …` form) to an output file.
class DecisionTreeClassifier implements ClassifierApi, LearnerApi<DataTree> {
  /// The model this classifier relies on; trained via [learnFromDataSet].
  DataTree tree;

  DecisionTreeClassifier({DataTree? tree}) : tree = tree ?? DataTree();

  @override
  ClassificationResultSet classify(DataSet toBeClassified) {
    if (!tree.isTrained) {
      throw StateError(
        'The decision tree must be trained before classification. Call learnFromDataSet first.',
      );
    }
    return tree.classify(toBeClassified);
  }

  @override
  DataTree learnFromDataSet(DataSet dataSet) {
    tree = DataTree.learn(dataSet);
    return tree;
  }

  /// Persists the trained tree as JSON to [path].
  void saveToFile(String path) {
    if (!tree.isTrained) {
      throw StateError('Cannot save an untrained decision tree.');
    }
    tree.saveToFile(path);
  }

  /// Trains a fresh classifier from a CSV file and saves the resulting tree.
  static DecisionTreeClassifier trainFromCsv(String csvPath, String outputPath) {
    final dataSet = DataSet.fromCsv(csvPath);
    return DecisionTreeClassifier()..trainFromDataSet(dataSet)..saveToFile(outputPath);
  }

  /// Trains from a [DataSet] and assigns the resulting model to this instance.
  DecisionTreeClassifier trainFromDataSet(DataSet dataSet) {
    tree = DataTree.learn(dataSet);
    return this;
  }

  /// Loads a previously saved tree and classifies [csvPath], writing verdicts to
  /// `outputPath` (one `stringify()` line per row) and returning the result set.
  static ClassificationResultSet classifyCsv(String csvPath, String outputPath) {
    final dataSet = DataSet.fromCsv(csvPath);
    final loaded = DataTree.loadFromFile(outputPath.replaceFirst('.json', ''));
    final resultSet = loaded.classify(dataSet);
    File(outputPath).writeAsStringSync(
      resultSet.results.map((r) => r.stringify()).join('\n'),
    );
    return resultSet;
  }

  /// Convenience: load a tree and classify an in-memory set.
  ClassificationResultSet classifyLoaded(DataTree model, DataSet toBeClassified) {
    return model.classify(toBeClassified);
  }
}
