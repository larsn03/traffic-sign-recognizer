import '../data/data_vector_set.dart';
import 'classification_result_set.dart';

/// The interface for all traffic sign classifiers.
abstract class ClassifierApi {
  ClassificationResultSet classify(DataVectorSet toBeClassified);
}
