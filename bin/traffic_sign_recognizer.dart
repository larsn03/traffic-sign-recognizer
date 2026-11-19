import 'dart:io';

import 'package:traffic_sign_recognizer/traffic_sign_recognizer.dart';

Future<void> main(List<String> args) async {
  final code = await run(args);
  exit(code);
}

/// Runs the CLI; returns the process exit code.
///
/// Argument parsing is a manual two-pass walk (no flag package), mirroring
/// the Go original:
///  * first pass extracts the standalone `-printfaulty` boolean flag;
///  * second pass dispatches on the mode flags
///    `-e <dir> <csv>`, `-knn <in> <out>`, `-dtree-train <train> <treeFile>`,
///    `-dtree-classify <tree> <test> <out>`;
///  * any other token must be an integer and sets the KNN k value;
///  * the first non-empty mode is dispatched.
Future<int> run(List<String> args) async {
  try {
    return await _dispatch(args);
  } on Object catch (e) {
    return _fail('$e');
  }
}

Future<int> _dispatch(List<String> args) async {
  // First pass: extract the standalone -printfaulty flag.
  var printfaulty = false;
  final filtered = <String>[];
  for (final arg in args) {
    if (arg == '-printfaulty') {
      printfaulty = true;
    } else {
      filtered.add(arg);
    }
  }

  // Second pass: dispatch on known mode flags.
  String? extractDir, extractCsv;
  String? classifyIn, classifyOut;
  String? dtTrain, dtTreeFile;
  String? dtTest, dtOut;
  var k = 5;

  var i = 0;
  while (i < filtered.length) {
    switch (filtered[i]) {
      case '-e':
        if (i + 2 >= filtered.length) {
          return _fail('missing arguments for -e');
        }
        extractDir = filtered[i + 1];
        extractCsv = filtered[i + 2];
        i += 3;
      case '-knn':
        if (i + 2 >= filtered.length) {
          return _fail('missing arguments for -knn');
        }
        classifyIn = filtered[i + 1];
        classifyOut = filtered[i + 2];
        i += 3;
      case '-dtree-train':
        if (i + 2 >= filtered.length) {
          return _fail('missing arguments for -dtree-train');
        }
        dtTrain = filtered[i + 1];
        dtTreeFile = filtered[i + 2];
        i += 3;
      case '-dtree-classify':
        if (i + 3 >= filtered.length) {
          return _fail('missing arguments for -dtree-classify');
        }
        dtTreeFile = filtered[i + 1];
        dtTest = filtered[i + 2];
        dtOut = filtered[i + 3];
        i += 4;
      default:
        final parsed = int.tryParse(filtered[i].trim());
        if (parsed == null) {
          return _fail('unknown argument "${filtered[i]}"');
        }
        k = parsed;
        i += 1;
    }
  }

  // Dispatch: the first non-empty mode wins.
  if (extractDir != null) {
    await Extractor().extractToCsv(extractDir, extractCsv!);
    return 0;
  }
  if (classifyIn != null) {
    final trainSet = DataVectorSet.fromCsv(classifyOut!);
    final testSet = DataVectorSet.fromCsv(classifyIn);
    final results =
        KnnClassifier(alreadyClassifiedData: trainSet, k: k)
            .classify(testSet);
    for (final result in results.results) {
      print(result.stringify());
    }
    if (printfaulty) logFaulty(results, 'classifying.log');
    _printAccuracy(results);
    return 0;
  }
  if (dtTrain != null) {
    final trainSet = DataVectorSet.fromCsv(dtTrain);
    final tree = DecisionTreeClassifier().learnFromDataSet(trainSet);
    tree.save(dtTreeFile!);
    print('Decision tree saved to $dtTreeFile');
    return 0;
  }
  if (dtTest != null) {
    final tree = DataTree.load(dtTreeFile!);
    final testSet = DataVectorSet.fromCsv(dtTest);
    final results = DecisionTreeClassifier(tree: tree).classify(testSet);
    writeResults(results, dtOut!);
    if (printfaulty) logFaulty(results, 'classifying.log');
    _printAccuracy(results);
    return 0;
  }
  return _fail('nothing to do: use -e, -knn, -dtree-train or -dtree-classify');
}

int _fail(String message) {
  stderr.writeln('error: $message');
  return 1;
}

void _printAccuracy(ClassificationResultSet results) {
  final rounded = (results.accuracy + 0.5).floor();
  print('Correctly classified: $rounded%');
}

/// Writes each result's faulty line (empty when correct) to [path],
/// creating the file fresh and skipping empty lines.
void logFaulty(ClassificationResultSet results, String path) {
  final buffer = StringBuffer();
  for (final result in results.results) {
    final line = result.stringifyIfFaulty();
    if (line.isNotEmpty) buffer.writeln(line);
  }
  final file = File(path);
  file.createSync(recursive: true);
  file.writeAsStringSync(buffer.toString(), mode: FileMode.write);
}

/// Writes every prediction, one per line, to [path].
void writeResults(ClassificationResultSet results, String path) {
  final buffer = StringBuffer();
  for (final result in results.results) {
    buffer.writeln(result.stringify());
  }
  final file = File(path);
  file.createSync(recursive: true);
  file.writeAsStringSync(buffer.toString(), mode: FileMode.write);
}
