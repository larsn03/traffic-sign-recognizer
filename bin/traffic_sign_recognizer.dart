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
///  * first pass extracts the standalone `-printfaulty`, `-z`, `-minmax`,
///    `-l1` and `-l2` boolean flags;
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
  // First pass: extract the standalone -printfaulty, -z, -minmax, -l1 and
  // -l2 flags.
  var printfaulty = false;
  var z = false;
  var minmax = false;
  var l1 = false;
  var l2 = false;
  final filtered = <String>[];
  for (final arg in args) {
    if (arg == '-printfaulty') {
      printfaulty = true;
    } else if (arg == '-z') {
      z = true;
    } else if (arg == '-minmax') {
      minmax = true;
    } else if (arg == '-l1') {
      l1 = true;
    } else if (arg == '-l2') {
      l2 = true;
    } else {
      filtered.add(arg);
    }
  }
  var standardizations = 0;
  if (z) standardizations++;
  if (minmax) standardizations++;
  if (l1) standardizations++;
  if (l2) standardizations++;
  if (standardizations > 1) {
    return _fail('-z, -minmax, -l1 and -l2 are mutually exclusive');
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
    if (z) {
      final params = trainSet.standardizeZ();
      testSet.standardizeZ(mean: params.$1, stddev: params.$2);
    } else if (minmax) {
      final params = trainSet.minMaxStandardize();
      testSet.minMaxStandardize(min: params.$1, max: params.$2);
    } else if (l1) {
      trainSet.l1Standardize();
      testSet.l1Standardize();
    } else if (l2) {
      trainSet.l2Standardize();
      testSet.l2Standardize();
    }
    final results = await KnnClassifier(alreadyClassifiedData: trainSet, k: k)
        .classify(testSet);
    final standardization =
        z ? 'z' : minmax ? 'minmax' : l1 ? 'l1' : l2 ? 'l2' : null;
    print('KNN | standardization: ${_standardizationName(standardization)}');
    if (printfaulty) logFaulty(results, 'classifying.log');
    _printAccuracy(results);
    return 0;
  }
  if (dtTrain != null) {
    final trainSet = DataVectorSet.fromCsv(dtTrain);
    String? standardization;
    List<double>? mean, stddev, min, max;
    if (z) {
      standardization = 'z';
      final params = trainSet.standardizeZ();
      mean = params.$1;
      stddev = params.$2;
    } else if (minmax) {
      standardization = 'minmax';
      final params = trainSet.minMaxStandardize();
      min = params.$1;
      max = params.$2;
    } else if (l1) {
      standardization = 'l1';
      trainSet.l1Standardize();
    } else if (l2) {
      standardization = 'l2';
      trainSet.l2Standardize();
    }
    final tree = DecisionTreeClassifier().learnFromDataSet(trainSet,
        standardization: standardization,
        mean: mean,
        stddev: stddev,
        min: min,
        max: max);
    tree.save(dtTreeFile!);
    print('Decision tree saved to $dtTreeFile');
    return 0;
  }
  if (dtTest != null) {
    final tree = DataTree.load(dtTreeFile!);
    final testSet = DataVectorSet.fromCsv(dtTest);
    final standardization = tree.standardization;
    if (z) {
      if (standardization != 'z') {
        return _fail(
            'tree $dtTreeFile was trained without -z; pass -z to classify with it');
      }
      testSet.standardizeZ(mean: tree.mean!, stddev: tree.stddev!);
    } else if (minmax) {
      if (standardization != 'minmax') {
        return _fail(
            'tree $dtTreeFile was trained without -minmax; pass -minmax to classify with it');
      }
      testSet.minMaxStandardize(min: tree.min!, max: tree.max!);
    } else if (l1) {
      if (standardization != 'l1') {
        return _fail(
            'tree $dtTreeFile was trained without -l1; pass -l1 to classify with it');
      }
      testSet.l1Standardize();
    } else if (l2) {
      if (standardization != 'l2') {
        return _fail(
            'tree $dtTreeFile was trained without -l2; pass -l2 to classify with it');
      }
      testSet.l2Standardize();
    } else if (standardization != null) {
      return _fail('tree $dtTreeFile was trained with standardization '
          '($standardization); pass -z, -minmax, -l1 or -l2 to classify with it');
    }
    final results =
        await DecisionTreeClassifier(tree: tree).classify(testSet);
    writeResults(results, dtOut!);
    print('Decision tree | standardization: '
        '${_standardizationName(tree.standardization)}');
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

/// Human-readable name of a standardization method (`null` = raw features).
String _standardizationName(String? s) => switch (s) {
      'z' => 'z-score',
      'minmax' => 'min-max',
      'l1' => 'L1 norm',
      'l2' => 'L2 norm',
      _ => 'none',
    };

void _printAccuracy(ClassificationResultSet results) {
  final rounded = (results.accuracy + 0.5).floor();
  print('Correctly classified: $rounded%');
  final ci = results.confidenceInterval;
  print('Correct: ${results.correctCount} / ${results.total}');
  print('Expected value: ${results.expectedValue.toStringAsFixed(4)}');
  print('Standard deviation: ${results.standardDeviation.toStringAsFixed(4)}');
  print('95% confidence interval: '
      '[${ci.$1.toStringAsFixed(4)}, ${ci.$2.toStringAsFixed(4)}]');
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
