import '../data/data_set.dart';
import 'classifier_api.dart';
import 'classification_result_set.dart';

/// A k-nearest-neighbour classifier, backed by already classified data.
class KnnClassifier implements ClassifierApi {
  DataSet alreadyClassifiedData;

  KnnClassifier({DataSet? alreadyClassifiedData})
      : alreadyClassifiedData = alreadyClassifiedData ?? DataSet();

  @override
  ClassificationResultSet classify(DataSet toBeClassified) {
    return ClassificationResultSet();
  }
}
