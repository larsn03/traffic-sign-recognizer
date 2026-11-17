import '../data/data_vector_set.dart';
import 'classifier_api.dart';
import 'classification_result_set.dart';

/// A k-nearest-neighbour classifier, backed by already classified data.
class KnnClassifier implements ClassifierApi {
  DataVectorSet alreadyClassifiedData;

  KnnClassifier({DataVectorSet? alreadyClassifiedData})
    : alreadyClassifiedData = alreadyClassifiedData ?? DataVectorSet();

  @override
  ClassificationResultSet classify(DataVectorSet toBeClassified) {
    return ClassificationResultSet();
  }
}
