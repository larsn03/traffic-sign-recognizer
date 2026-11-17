import '../data/data_vector_set.dart';

/// The interface for anything that can learn from a data set.
abstract class LearnerApi<DataStructure> {
  DataStructure learnFromDataSet(DataVectorSet dataSet);
}
