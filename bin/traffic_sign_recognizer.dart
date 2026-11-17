import 'dart:io';

import 'package:traffic_sign_recognizer/traffic_sign_recognizer.dart';

/// Builds a labelled feature vector for a small demo set.
///
/// The first column holds the number of different colours — the one signal the
/// extractor emphasises — while column 25 carries a white-frame fraction used as
/// secondary signal, so both classifiers have something real to learn from.
DataVector makeSample(String label, String path, double colors, double signal) {
  final features = List<double>.filled(DataVector.featureCount, 0.0);
  features[0] = colors;
  features[25] = signal;
  return DataVector.fromNumericFeatures(label, path, features);
}

/// Creates the training examples: two tightly grouped classes of six vectors.
DataSet buildTrainingData() {
  final data = <DataVector>[];
  for (final colors in [1, 2]) {
    for (final signal in [0, 1]) {
      data.add(makeSample('Sign_A', 'a_$colors.jpg', colors.toDouble(), signal.toDouble()));
    }
  }
  for (final colors in [7, 8]) {
    for (final signal in [6, 7]) {
      data.add(makeSample('Sign_B', 'b_$colors.jpg', colors.toDouble(), signal.toDouble()));
    }
  }
  return DataSet(data: data);
}

/// A held-out test set to evaluate the trained models.
DataSet buildTestData() {
  final data = <DataVector>[];
  data.add(makeSample('Sign_A', 'test_a.jpg', 1.5, 0.5));
  data.add(makeSample('Sign_B', 'test_b.jpg', 8.0, 6.5));
  return DataSet(data: data);
}

void report(String title, ClassificationResultSet result) {
  final buffer = StringBuffer()
    ..writeln('$title')
    ..writeln('  total: ${result.total}')
    ..writeln('  correct: ${result.correctCount}')
    ..writeln('  accuracy: ${result.accuracy.toStringAsFixed(2)}%')
    ..writeln('  faulty: ${result.faultyCount}');

  final lines = result.results.map((r) {
    return '    [${r.correctlyClassified ? "ok" : "faulty"}] ${r.stringify()}';
  }).join('\n');
  buffer.writeln(lines);

  print(buffer.toString());
  File('results.csv').writeAsStringSync(result.toCsvString());
  for (final faulty in result.faulty) {
    File('faulty.log')
        .writeAsStringSync('${faulty.stringify()}\n', mode: FileMode.append);
  }
}

/// Writes each misclassified verdict to `classifying.log`, one per line.
void _logFaulty(ClassificationResultSet result) {
  final buffer = StringBuffer();
  for (final faulty in result.faulty) {
    buffer.writeln(faulty.stringify());
  }
  File('classifying.log').writeAsStringSync(buffer.toString());
}

/// Runs the KNN classifier over the held-out [testCsv] using a labelled
/// set from [trainCsv], writing predictions to `results.csv` and printing the
/// accuracy. A trailing `-printfaulty` flag also dumps misclassified rows.
void _classifyKnn(String testCsv, String trainCsv, int k, bool printFaulty) {
  final testSet = DataSet.fromCsv(testCsv);
  final trainSet = DataSet.fromCsv(trainCsv);

  final knn = KnnClassifier(alreadyClassifiedData: trainSet, k: k);
  final result = knn.classify(testSet);

  File('results.csv').writeAsStringSync(result.toCsvString());
  print(
    'Correctly classified: ${result.accuracy.toStringAsFixed(0)}%',
  );
  if (printFaulty) {
    _logFaulty(result);
    print('Faulty images logged to classifying.log');
  }
}

/// Extracts features from [src] — a directory of images (classes are the
/// subfolder names), a single image file, or a CSV — and writes the resulting
/// feature vectors to [outCsv].
void _extract(String src, String outCsv) async {
  final dataSet = await const ImageExtractor().extract(src);
  File(outCsv).writeAsStringSync(dataSet.toCsvString());
  print('Extraction complete: wrote ${dataSet.length} rows to $outCsv');
}

/// Trains a decision tree on the labelled CSV in [trainCsv] and saves it as JSON.
void _trainTree(String trainCsv, String treeJson) {
  final dataSet = DataSet.fromCsv(trainCsv);
  final tree = DecisionTreeClassifier().learnFromDataSet(dataSet);
  File(treeJson).writeAsStringSync(tree.toJsonString());
  print('Decision tree trained and saved to $treeJson');
}

/// Classifies [testCsv] with the persisted tree at [jsonPath], writing verdicts
/// to `results_tree.csv` (or a custom output path) and printing accuracy.
void _classifyTree(String jsonPath, String testCsv, String? outCsv, bool printFaulty) {
  final model = DataTree.loadFromFile(jsonPath);
  final dataSet = DataSet.fromCsv(testCsv);

  final dt = DecisionTreeClassifier();
  final result = dt.classifyLoaded(model, dataSet);

  File(outCsv ?? 'results_tree.csv').writeAsStringSync(result.toCsvString());
  print(
    'Correctly classified: ${result.accuracy.toStringAsFixed(0)}%',
  );
  if (printFaulty) {
    _logFaulty(result);
    print('Faulty images logged to classifying.log');
  }
}

/// Runs the self-contained sample pipeline in-memory and prints a report for
/// each model, writing results.csv, faulty.log and tree.json.
void _runDemo() {
  print('Traffic Sign Recognizer — sample pipeline\n');

  final training = buildTrainingData();
  final test = buildTestData();

  // KNN classifier with feature standardization before distance queries.
  final knn = KnnClassifier(alreadyClassifiedData: training, k: 3);
  report('KNN classification (k=3)', knn.classify(test));

  // Decision tree: learn, classify, then persist and reload from disk.
  final dt = DecisionTreeClassifier();
  final trainedTree = dt.learnFromDataSet(training);
  report(
    'Decision-tree classification',
    dt.classify(test),
  );
  print('\nTrained node shape: ${trainedTree.toJsonString().length} bytes of JSON');

  File('tree.json').writeAsStringSync(trainedTree.toJsonString());
  final loadedTree = DataTree.loadFromFile('tree.json');
  report(
    'Decision-tree classification (from tree.json)',
    loadedTree.classify(test),
  );

  print('\nOutputs written: results.csv, faulty.log, tree.json');
}

/// Parses command-line args like the Go CLI and dispatches to one mode.
///
/// Recognised flags (positional args are consumed greedily per flag):
///
///   -e <src> <outCsv>          Extract feature data from [src] into a CSV.
///   -knn <testCsv> <trainCsv>  Classify [testCsv] with the labelled [trainCsv];
///                              an optional trailing integer sets k.
///   -dtree-train <csv> <json>  Train a decision tree and save it as JSON.
///   -dtree-classify <json> <testCsv> <outCsv>
///                              Classify [testCsv] with a saved tree.
///
/// A standalone `-printfaulty` flag (anywhere in the args) makes each classify
/// mode also dump its misclassified verdicts to `classifying.log`. With no
/// recognised flag, the synthetic sample pipeline is run instead.
int main(List<String> args) {
  final printFaulty = args.contains('-printfaulty');
  var remaining = <String>[for (final a in args) if (a != '-printfaulty') a];

  String? dirIn, csvOut; // -e
  String? testCsv, trainCsv; // -knn
  int k = 5; // neighbours for KNN
  String? dtTrainCsv, treeJsonFile; // -dtree-train
  String? jsonPath, dtTestCsv, dtOutCsv; // -dtree-classify

  var i = 0;
  while (i < remaining.length) {
    switch (remaining[i]) {
      case '-e':
        if (i + 2 < remaining.length) { dirIn = remaining[i + 1]; csvOut = remaining[i + 2]; }
        i += 3; break;
      case '-knn':
        if (i + 2 < remaining.length) { testCsv = remaining[i + 1]; trainCsv = remaining[i + 2]; }
        i += 3; break;
      case '-dtree-train':
        if (i + 2 < remaining.length) { dtTrainCsv = remaining[i + 1]; treeJsonFile = remaining[i + 2]; }
        i += 3; break;
      case '-dtree-classify':
        if (i + 3 < remaining.length) {
          jsonPath = remaining[i + 1];
          dtTestCsv = remaining[i + 2];
          dtOutCsv = remaining[i + 3];
        }
        i += 4; break; // flag + three path args
      default:
        final parsed = int.tryParse(remaining[i]);
        if (parsed != null) {
          k = parsed;
        } else {
          print('unknown argument "${remaining[i]}"');
          return 1;
        }
        i++; break;
    }
  }

  if (dirIn != null && csvOut != null) {
    _extract(dirIn, csvOut);
  } else if (testCsv != null && trainCsv != null) {
    _classifyKnn(testCsv, trainCsv, k, printFaulty);
  } else if (dtTrainCsv != null && treeJsonFile != null) {
    _trainTree(dtTrainCsv, treeJsonFile);
  } else if (jsonPath != null && dtTestCsv != null) {
    _classifyTree(jsonPath, dtTestCsv, dtOutCsv, printFaulty);
  } else {
    _runDemo();
  }

  return 0;
}
