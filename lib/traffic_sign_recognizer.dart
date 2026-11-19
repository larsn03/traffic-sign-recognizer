/// Traffic Sign Recognizer — a Dart port of the Go reference project.
///
/// Exposes the data model, feature extraction and the KNN / decision-tree
/// classifiers as a single library.
library;

export 'classifier/classifier_api.dart';
export 'classifier/classification_result.dart';
export 'classifier/classification_result_set.dart';
export 'classifier/decision_tree_classifier.dart';
export 'classifier/knn_classifier.dart';
export 'classifier/learner_api.dart';

export 'data/data_mask.dart';
export 'data/data_tree.dart';
export 'data/data_vector.dart';
export 'data/data_vector_set.dart';

export 'extraction/colors.dart';
export 'extraction/extractor.dart';
export 'extraction/features.dart';
