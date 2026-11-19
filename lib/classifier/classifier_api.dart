import '../data/data_vector_set.dart';
import 'classification_result_set.dart';

/// The interface for all traffic sign classifiers.
abstract class ClassifierApi {
  Future<ClassificationResultSet> classify(DataVectorSet toBeClassified);
}
