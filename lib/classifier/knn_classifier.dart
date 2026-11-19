import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import '../data/data_mask.dart';
import '../data/data_vector.dart';
import '../data/data_vector_set.dart';
import 'classifier_api.dart';
import 'classification_result.dart';
import 'classification_result_set.dart';

/// A k-nearest-neighbour classifier, backed by already classified data.
///
/// Mirrors the Go project's `dataclassification` package: z-score
/// standardization fitted on the training set only (population standard
/// deviation, zero-variance columns mapped to 0), Euclidean distance,
/// majority vote with alphabetical tie-breaking.
///
/// Every test row is classified independently of the others, so large test
/// sets are distributed across a pool of isolates with dynamic work
/// distribution: each worker is handed the next row as soon as it finishes
/// its current one (the same scheme as the extractor). Results are stored
/// by row index, so the output order matches the input order and every
/// prediction is bit-identical to a sequential run.
class KnnClassifier implements ClassifierApi {
  DataVectorSet alreadyClassifiedData;

  /// Number of neighbours; values <= 0 fall back to the default of 5.
  final int k;

  /// Test sets with fewer rows than this are classified on the main
  /// isolate; spawning a worker pool is not worth it below this size.
  final int minRowsForIsolates;

  /// When true, features are first multiplied by their [_mask] family
  /// factor.
  final bool applyMask;

  /// When true, features are z-score standardized (fit on the training
  /// set), after the optional data-mask scaling.
  final bool applyZscore;

  /// The data mask used when [applyMask] is true.
  final DataMask _mask;

  KnnClassifier({
    DataVectorSet? alreadyClassifiedData,
    this.k = 5,
    int? minRowsForIsolates,
    this.applyMask = false,
    this.applyZscore = true,
    DataMask? mask,
  })  : alreadyClassifiedData = alreadyClassifiedData ?? DataVectorSet(),
        minRowsForIsolates = minRowsForIsolates ?? _defaultMinRowsForIsolates,
        _mask = mask ?? DataMask.identity();

  /// Default row count below which classification stays single-isolate.
  static const int _defaultMinRowsForIsolates = 1000;

  @override
  Future<ClassificationResultSet> classify(DataVectorSet toBeClassified) async {
    final kk = k <= 0 ? 5 : k;
    final train = alreadyClassifiedData.data;
    final test = toBeClassified.data;
    if (test.isEmpty) {
      return ClassificationResultSet();
    }

    final numFeatures = DataVector.featureCount;

    final trainLabels = [for (final vector in train) vector.trafficSign];
    // Feature pipeline: raw features, then optionally the data mask, then
    // optionally z-score standardization (fit on the masked training set).
    var trainFlat = _rawFlat(train, numFeatures);
    var testFlat = _rawFlat(test, numFeatures);
    if (applyMask) {
      trainFlat = _scaleFlat(trainFlat, _mask, numFeatures);
      testFlat = _scaleFlat(testFlat, _mask, numFeatures);
    }
    if (applyZscore) {
      final (mean, stddev) = _fitStandardization(trainFlat, numFeatures);
      trainFlat =
          _standardizeFlat(trainFlat, mean, stddev, numFeatures);
      testFlat = _standardizeFlat(testFlat, mean, stddev, numFeatures);
    }

    final results = List<ClassificationResult?>.filled(test.length, null);
    if (test.length < minRowsForIsolates) {
      // Small test set: stay on the main isolate.
      for (var i = 0; i < test.length; i++) {
        results[i] = ClassificationResult(
          path: test[i].picturePath,
          shouldBe: test[i].trafficSign,
          predicted: _classifyOneFlat(
              testFlat, i, trainFlat, trainLabels, kk, numFeatures),
        );
      }
    } else {
      await _classifyParallel(
          test, results, testFlat, trainFlat, trainLabels, kk, numFeatures);
    }

    var correct = 0;
    final finalResults = <ClassificationResult>[];
    for (final result in results) {
      final r = result!;
      finalResults.add(r);
      if (r.isCorrect) correct++;
    }
    return ClassificationResultSet(
      results: finalResults,
      accuracy: correct / test.length * 100,
    );
  }

  /// Copies the raw features of [vectors] into a flat, tightly packed
  /// buffer.
  static Float64List _rawFlat(List<DataVector> vectors, int numFeatures) {
    final out = Float64List(vectors.length * numFeatures);
    for (var i = 0; i < vectors.length; i++) {
      out.setRange(i * numFeatures, i * numFeatures + numFeatures,
          vectors[i].features());
    }
    return out;
  }

  /// Multiplies every feature of the flat buffer [flat] by its data-mask
  /// family factor.
  static Float64List _scaleFlat(
      Float64List flat, DataMask mask, int numFeatures) {
    final n = flat.length ~/ numFeatures;
    final out = Float64List(flat.length);
    for (var i = 0; i < n; i++) {
      final base = i * numFeatures;
      out.setRange(base, base + numFeatures,
          mask.apply(flat.sublist(base, base + numFeatures)));
    }
    return out;
  }

  /// Fits the z-score parameters (mean, population standard deviation) on
  /// the training rows of [trainFlat]; the operation order matches the
  /// original per-vector accumulation exactly.
  static (Float64List, Float64List) _fitStandardization(
      Float64List trainFlat, int numFeatures) {
    final n = trainFlat.length ~/ numFeatures;
    final mean = Float64List(numFeatures);
    for (var i = 0; i < n; i++) {
      final base = i * numFeatures;
      for (var j = 0; j < numFeatures; j++) {
        mean[j] += trainFlat[base + j];
      }
    }
    for (var j = 0; j < numFeatures; j++) {
      mean[j] /= n;
    }
    final variance = Float64List(numFeatures);
    for (var i = 0; i < n; i++) {
      final base = i * numFeatures;
      for (var j = 0; j < numFeatures; j++) {
        final d = trainFlat[base + j] - mean[j];
        variance[j] += d * d;
      }
    }
    final stddev = Float64List(numFeatures);
    for (var j = 0; j < numFeatures; j++) {
      stddev[j] = math.sqrt(variance[j] / n);
    }
    return (mean, stddev);
  }

  /// Applies the z-score standardization to the flat buffer [flat]
  /// (zero-variance columns map to 0).
  static Float64List _standardizeFlat(
      Float64List flat, Float64List mean, Float64List stddev, int numFeatures) {
    final out = Float64List(flat.length);
    for (var i = 0; i < flat.length; i++) {
      final j = i % numFeatures;
      final sd = stddev[j];
      out[i] = sd == 0 ? 0 : (flat[i] - mean[j]) / sd;
    }
    return out;
  }

  /// Euclidean distance from test row [row] to every training vector,
  /// majority vote over the k nearest; ties broken by alphabetical
  /// (lexicographically smallest) label.
  static String _classifyOneFlat(
    Float64List testFlat,
    int row,
    Float64List trainFlat,
    List<String> labels,
    int k,
    int numFeatures,
  ) {
    final n = trainFlat.length ~/ numFeatures;
    final queryBase = row * numFeatures;
    final distances = List<double>.generate(n, (i) {
      final base = i * numFeatures;
      var sum = 0.0;
      for (var j = 0; j < numFeatures; j++) {
        final d = testFlat[queryBase + j] - trainFlat[base + j];
        sum += d * d;
      }
      return math.sqrt(sum);
    });
    final order = List<int>.generate(n, (i) => i)
      ..sort((a, b) => distances[a].compareTo(distances[b]));
    final votes = <String, int>{};
    for (var m = 0; m < k; m++) {
      final label = labels[order[m]];
      votes[label] = (votes[label] ?? 0) + 1;
    }
    var best = '';
    var bestVotes = -1;
    for (final label in votes.keys.toList()..sort()) {
      final count = votes[label]!;
      if (count > bestVotes) {
        bestVotes = count;
        best = label;
      }
    }
    return best;
  }

  /// Distributes the test rows across a pool of isolates using dynamic work
  /// distribution: workers register themselves, receive one row index at a
  /// time, and are fed the next row as soon as they report a result, so
  /// the pool stays fully busy until every row is done.
  Future<void> _classifyParallel(
    List<DataVector> test,
    List<ClassificationResult?> results,
    Float64List testFlat,
    Float64List trainFlat,
    List<String> trainLabels,
    int k,
    int numFeatures,
  ) async {
    final workers = math.min(_workerCount(), test.length);
    final port = ReceivePort();
    final done = Completer<void>();
    final workerPorts = <SendPort>[];
    var nextIndex = 0;
    var remaining = test.length;

    // Hand the next queued row to [workerPort]; a no-op once every row has
    // been dispatched.
    void dispatch(SendPort workerPort) {
      if (nextIndex < test.length) {
        workerPort.send(nextIndex++);
      }
    }

    final subscription = port.listen((dynamic message) {
      if (message is SendPort) {
        // A worker registered; give it its first row.
        workerPorts.add(message);
        dispatch(message);
      } else {
        final list = message as List<dynamic>;
        final index = list[0] as int;
        final predicted = list[1] as String;
        results[index] = ClassificationResult(
          path: test[index].picturePath,
          shouldBe: test[index].trafficSign,
          predicted: predicted,
        );
        remaining--;
        final workerPort = list[2] as SendPort;
        if (remaining > 0) {
          // This worker just freed up; feed it the next row.
          dispatch(workerPort);
        } else {
          // Everything is done; shut every worker down.
          for (final sp in workerPorts) {
            sp.send(null);
          }
          done.complete();
        }
      }
    });

    for (var i = 0; i < workers; i++) {
      Isolate.spawn(_knnWorkerMain, <dynamic>[
        port.sendPort,
        trainFlat,
        trainLabels,
        testFlat,
        k,
        numFeatures,
      ]);
    }
    await done.future;
    await subscription.cancel();
    port.close();
  }
}

/// Worker entry point: announces itself, then classifies row indices until
/// it receives the shutdown sentinel (null).
void _knnWorkerMain(List<dynamic> args) {
  final resultPort = args[0] as SendPort;
  final trainFlat = args[1] as Float64List;
  final trainLabels = args[2] as List<String>;
  final testFlat = args[3] as Float64List;
  final k = args[4] as int;
  final numFeatures = args[5] as int;

  final port = ReceivePort();
  resultPort.send(port.sendPort);
  port.listen((dynamic message) {
    if (message == null) {
      port.close();
      return;
    }
    final index = message as int;
    final predicted = KnnClassifier._classifyOneFlat(
        testFlat, index, trainFlat, trainLabels, k, numFeatures);
    resultPort.send(<dynamic>[index, predicted, port.sendPort]);
  });
}

/// Chooses the worker count: honour the `TSR_WORKERS` environment variable
/// if set, else default to the logical processor count.
int _workerCount() {
  final fromEnv = int.tryParse(Platform.environment['TSR_WORKERS'] ?? '');
  if (fromEnv != null && fromEnv > 0) return fromEnv;
  return math.max(1, _processorCount());
}

/// The number of logical CPU threads, floored to at least 1.
int _processorCount() {
  try {
    final result = Process.runSync('nproc', []);
    final count = int.tryParse(result.stdout.toString().trim());
    if (count != null && count > 0) return count;
  } catch (_) {
    // fall through to the default
  }
  return 1;
}
