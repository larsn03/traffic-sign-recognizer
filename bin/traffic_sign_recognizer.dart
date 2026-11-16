import 'package:traffic_sign_recognizer/traffic_sign_recognizer.dart';

/// Prints a short summary of the given classifier.
String knnclassifierToString(KnnClassifier classifier) {
  return 'KnnClassifier{alreadyClassifiedData: ${classifier.alreadyClassifiedData}}';
}

void main() {
  print('Hello and welcome!');

  for (int i = 1; i <= 5; i++) {
    print('i = $i');
  }

  // Demonstrate the converted classifier & extraction code.
  final dataSet = DataSet();
  final extractor = Extractor();
  final alreadyClassified = extractor.extractToCsv('data/traffic_signs.csv');

  final knn = KnnClassifier(alreadyClassifiedData: alreadyClassified);
  final decisionTree = DecisionTreeClassifier();
  final tree = decisionTree.learnFromDataSet(dataSet);

  print('knn classifier: ${knnclassifierToString(knn)}');
  print('decision tree: $tree');
}
