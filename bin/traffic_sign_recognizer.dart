import 'package:traffic_sign_recognizer/traffic_sign_recognizer.dart';

/// Prints a short summary of the given classifier.
String knnclassifierToString(KnnClassifier classifier) {
  return 'KnnClassifier{alreadyClassifiedData: ${classifier.alreadyClassifiedData}}';
}

void main() {
  print('Hello and welcome!');
}
