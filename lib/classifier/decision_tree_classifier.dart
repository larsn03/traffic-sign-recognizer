import '../data/data_set.dart';
import '../data/data_tree.dart';
import 'classifier_api.dart';
import 'classification_result_set.dart';
import 'learner_api.dart';

/// A classifier that uses a decision tree, which it can learn from data.
class DecisionTreeClassifier implements ClassifierApi, LearnerApi<DataTree> {
  DataTree tree;

  DecisionTreeClassifier({DataTree? tree}) : tree = tree ?? DataTree();

  @override
  ClassificationResultSet classify(DataSet toBeClassified) {
    return ClassificationResultSet();
  }

  @override
  DataTree learnFromDataSet(DataSet dataSet) {
    return tree;
  }
}
