import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;

import 'package:image/image.dart';
import 'package:path/path.dart' as p;

import '../data/data_vector.dart';
import '../data/data_vector_set.dart';
import 'features.dart';

/// Extracts data from a directory of labelled images into a
/// [DataVectorSet], optionally writing the CSV representation.
///
/// Mirrors the Go project's `data_extraction` package: [pathToData] contains
/// one subfolder per class label; images may be nested at arbitrary depth.
///
/// [extractSplit] is the main entry point: it shuffles all images with a
/// printed seed, puts the first 60% into a training CSV and the rest into a
/// classification CSV.
///
/// Parallelism: extraction runs on a pool of isolates. Because the Dart VM
/// caps the number of isolates that run in parallel inside one process
/// (the cap is derived from the heap space available to the process),
/// [extractSplit] splits large job lists into contiguous shards and runs
/// one child process per shard — each with its own isolate pool — so the
/// whole machine's core count can be saturated. Shard processes are
/// signalled via the `TSR_SHARDS`/`TSR_SHARD`/`TSR_SEED` environment
/// variables; the worker count can be overridden with `TSR_WORKERS` (used
/// by the orchestrator and for debugging).
class Extractor {
  /// Supported image extensions (case-insensitive).
  static const Set<String> supportedExtensions = {
    'bmp',
    'png',
    'jpg',
    'jpeg',
    'gif',
    'tif',
    'tiff',
  };

  /// Isolates per process in multi-process mode. The Dart VM only runs a
  /// limited number of isolates in parallel per process (roughly 14-15 on a
  /// typical machine); staying in this range keeps every worker fully
  /// utilised.
  static const int _workersPerProcess = 14;

  /// Below this many jobs the child-process startup overhead is not
  /// worth it; extraction stays in a single process.
  static const int _minJobsForMultiProcess = 500;

  /// Walks [pathToData] and extracts one vector per image (no CSV written).
  ///
  /// When [shard] and [shards] are given, only the contiguous chunk of
  /// jobs belonging to that shard is processed (used by the multi-process
  /// orchestration in [extractToCsv]).
  Future<DataVectorSet> extract(
    String pathToData, {
    int? shard,
    int? shards,
  }) async {
    final jobs = _collectJobs(pathToData, shard: shard, shards: shards);
    final vectors = await _extractParallel(jobs);
    final data = <DataVector>[for (final vector in vectors) ?vector];
    return DataVectorSet(data: data);
  }

  /// Extracts and writes the CSV to [pathToCsv].
  Future<DataVectorSet> extractToCsv(String pathToData, String pathToCsv) async {
    final jobs = _collectJobs(pathToData);
    final vectors = await _extractParallel(jobs);
    final set = DataVectorSet(
        data: [for (final vector in vectors) ?vector]);
    set.toCsv(pathToCsv);
    return set;
  }

  /// The fraction of all images that become training data; the rest
  /// becomes classification data.
  static const double trainFraction = 0.6;

  /// Extracts every image under [root] and splits them into two CSVs: the
  /// first [trainFraction] (after a seeded shuffle) go to [trainCsv], the
  /// rest to [testCsv].
  ///
  /// Every call reshuffles: when [seed] is not supplied a fresh random
  /// seed is used, and the seed is printed so the split can be
  /// reproduced. A process that is itself a shard (signalled by the
  /// `TSR_SHARDS`/`TSR_SHARD`/`TSR_SEED` environment variables) only
  /// extracts its slice of the same deterministic shuffle and writes to
  /// the shard CSVs it was given.
  Future<int> extractSplit(
      String root, String trainCsv, String testCsv,
      {int? seed}) async {
    final env = Platform.environment;
    final shards = int.tryParse(env['TSR_SHARDS'] ?? '');
    final shard = int.tryParse(env['TSR_SHARD'] ?? '');
    if (shards != null && shards > 1 && shard != null) {
      final s = int.parse(env['TSR_SEED']!);
      await _extractSplitShard(root, s, shard, shards, trainCsv, testCsv);
      return s;
    }

    final s = seed ?? math.Random().nextInt(1 << 30);
    print('extraction seed: $s');
    final jobs = _collectJobs(root);
    final totalWorkers = _defaultWorkerCount();
    final numProcs = totalWorkers <= _workersPerProcess
        ? 1
        : (totalWorkers + _workersPerProcess - 1) ~/ _workersPerProcess;
    if (numProcs > 1 && jobs.length >= _minJobsForMultiProcess) {
      await _extractSplitMultiProcess(root, trainCsv, testCsv, s, numProcs);
    } else {
      await _extractSplitInProcess(jobs, s, trainCsv, testCsv);
    }
    return s;
  }

  /// Shuffles [jobs] with [seed] and splits them into (training, test)
  /// lists: the first [trainFraction] become training jobs, the rest test
  /// jobs. The jobs are reindexed so parallel results come back in subset
  /// order.
  (List<_ExtractionJob>, List<_ExtractionJob>) _splitJobs(
      List<_ExtractionJob> jobs, int seed) {
    final order = List<int>.generate(jobs.length, (i) => i)
      ..shuffle(math.Random(seed));
    final trainCount = (jobs.length * trainFraction).floor();
    final trainJobs = <_ExtractionJob>[];
    final testJobs = <_ExtractionJob>[];
    for (var rank = 0; rank < order.length; rank++) {
      final job = jobs[order[rank]];
      final subset = rank < trainCount ? trainJobs : testJobs;
      subset.add(_ExtractionJob(subset.length, job.label, job.relativePath,
          job.absolutePath));
    }
    return (trainJobs, testJobs);
  }

  /// Splits in memory and extracts both subsets in this process.
  Future<void> _extractSplitInProcess(List<_ExtractionJob> jobs, int seed,
      String trainCsv, String testCsv) async {
    final (trainJobs, testJobs) = _splitJobs(jobs, seed);
    final trainVectors = await _extractParallel(trainJobs);
    DataVectorSet(data: [for (final v in trainVectors) ?v]).toCsv(trainCsv);
    final testVectors = await _extractParallel(testJobs);
    DataVectorSet(data: [for (final v in testVectors) ?v]).toCsv(testCsv);
  }

  /// Extracts this shard's contiguous slice of the shared shuffle and
  /// writes its training rows to [trainCsv] and its test rows to
  /// [testCsv] (both shard CSVs in a temp dir).
  Future<void> _extractSplitShard(String root, int seed, int shard, int shards,
      String trainCsv, String testCsv) async {
    final jobs = _collectJobs(root);
    final order = List<int>.generate(jobs.length, (i) => i)
      ..shuffle(math.Random(seed));
    final trainCount = (jobs.length * trainFraction).floor();
    final per = jobs.length ~/ shards;
    final start = shard * per;
    final end = shard == shards - 1 ? jobs.length : start + per;
    final trainJobs = <_ExtractionJob>[];
    final testJobs = <_ExtractionJob>[];
    for (var rank = start; rank < end; rank++) {
      final job = jobs[order[rank]];
      final subset = rank < trainCount ? trainJobs : testJobs;
      subset.add(_ExtractionJob(subset.length, job.label, job.relativePath,
          job.absolutePath));
    }
    final trainVectors = await _extractParallel(trainJobs);
    DataVectorSet(data: [for (final v in trainVectors) ?v]).toCsv(trainCsv);
    final testVectors = await _extractParallel(testJobs);
    DataVectorSet(data: [for (final v in testVectors) ?v]).toCsv(testCsv);
  }

  /// Runs one child process per shard; each child reproduces the seeded
  /// shuffle, extracts its slice, and writes shard CSVs for both subsets.
  /// The shard CSVs are then merged into [trainCsv] and [testCsv].
  Future<void> _extractSplitMultiProcess(String root, String trainCsv,
      String testCsv, int seed, int numProcs) async {
    final scriptPath = Platform.script.toFilePath();
    final baseArgs = <String>[
      if (scriptPath != Platform.resolvedExecutable) scriptPath,
      '-e',
      root,
    ];
    final tempDir = Directory.systemTemp.createTempSync('tsr-split-');
    final trainShardCsvs = <String>[];
    final testShardCsvs = <String>[];
    final exitCodes = <Future<int>>[];
    try {
      for (var i = 0; i < numProcs; i++) {
        final trainShardCsv = p.join(tempDir.path, 'train_$i.csv');
        final testShardCsv = p.join(tempDir.path, 'test_$i.csv');
        trainShardCsvs.add(trainShardCsv);
        testShardCsvs.add(testShardCsv);
        final process = await Process.start(
          Platform.resolvedExecutable,
          [...baseArgs, trainShardCsv, testShardCsv],
          environment: {
            ...Platform.environment,
            'TSR_SHARDS': '$numProcs',
            'TSR_SHARD': '$i',
            'TSR_SEED': '$seed',
            'TSR_WORKERS': '$_workersPerProcess',
          },
        );
        exitCodes.add(process.exitCode);
      }
      for (var i = 0; i < numProcs; i++) {
        final code = await exitCodes[i];
        if (code != 0) {
          throw ProcessException(Platform.resolvedExecutable, baseArgs,
              'shard $i exited with code $code', code);
        }
      }
      _mergeCsvs(trainShardCsvs, trainCsv);
      _mergeCsvs(testShardCsvs, testCsv);
    } finally {
      tempDir.deleteSync(recursive: true);
    }
  }

  /// Concatenates the shard CSVs (shards were assigned contiguous slices in
  /// sorted order) into [pathToCsv], keeping a single header row. The
  /// output matches [DataVectorSet.toCsv] byte for byte (no trailing
  /// newline).
  static void _mergeCsvs(List<String> shardCsvs, String pathToCsv) {
    final lines = <String>[];
    var headerWritten = false;
    for (final shardCsv in shardCsvs) {
      final shardLines = File(shardCsv).readAsLinesSync();
      if (shardLines.isEmpty) continue;
      if (!headerWritten) {
        lines.add(shardLines.first);
        headerWritten = true;
      }
      lines.addAll(shardLines.skip(1));
    }
    final file = File(pathToCsv);
    file.createSync(recursive: true);
    file.writeAsStringSync(lines.join('\n'));
  }
}

class _ExtractionJob {
  final int index;
  final String label;
  final String relativePath;
  final String absolutePath;
  _ExtractionJob(
      this.index, this.label, this.relativePath, this.absolutePath);
}

List<_ExtractionJob> _collectJobs(
  String root, {
  int? shard,
  int? shards,
}) {
  final rootDir = Directory(root);
  if (!rootDir.existsSync()) {
    throw FileSystemException('extraction root not found', root);
  }
  final files = rootDir
      .listSync(recursive: true, followLinks: false)
      .whereType<File>()
      .map((f) => f.path)
      .where((path) =>
          Extractor.supportedExtensions
              .contains(p.extension(path).toLowerCase().substring(1)))
      .toList()
    ..sort();

  // Multi-process sharding: a shard receives a contiguous slice of the
  // sorted file list, so concatenating the shard CSVs in order
  // reconstructs the original ordering.
  var start = 0;
  var end = files.length;
  if (shards != null && shards > 1 && shard != null) {
    if (shard < 0 || shard >= shards) return const [];
    final per = files.length ~/ shards;
    start = shard * per;
    end = shard == shards - 1 ? files.length : start + per;
  }

  return [
    for (var i = start; i < end; i++)
      _ExtractionJob(
        i - start,
        p.split(p.relative(files[i], from: root)).first,
        p.relative(files[i], from: root),
        files[i],
      ),
  ];
}

/// Distributes [jobs] across a pool of [workerCount] isolates using
/// dynamic work distribution: each worker is handed the next job as soon as
/// it finishes its current one, so the pool self-balances and stays fully
/// busy until every job is done (no static partition, no idle tail).
///
/// Results are stored by job index so ordering matches the input order
/// after all workers finish. A null outcome means the image was skipped
/// (a warning was printed to stderr by the worker).
Future<List<DataVector?>> _extractParallel(
    List<_ExtractionJob> jobs, {
    int? workerCount,
  }) async {
  if (jobs.isEmpty) return <DataVector?>[];
  final workers = workerCount ?? _defaultWorkerCount();
  final results = List<DataVector?>.filled(jobs.length, null);
  final port = ReceivePort();
  final done = Completer<void>();
  final workerPorts = <SendPort>[];
  var nextIndex = 0;
  var remaining = jobs.length;

  // Hand the next queued job to [workerPort]; a no-op once everything is
  // dispatched.
  void dispatch(SendPort workerPort) {
    if (nextIndex < jobs.length) {
      workerPort.send(jobs[nextIndex++]);
    }
  }

  final subscription = port.listen((dynamic message) {
    if (message is SendPort) {
      // A worker registered; give it its first job.
      workerPorts.add(message);
      dispatch(message);
    } else {
      final list = message as List<dynamic>;
      results[list[0] as int] = list[1] as DataVector?;
      remaining--;
      final workerPort = list[2] as SendPort;
      if (remaining > 0) {
        // This worker just freed up; feed it the next job.
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
    Isolate.spawn(_workerMain, port.sendPort);
  }
  await done.future;
  await subscription.cancel();
  port.close();
  return results;
}

/// Chooses the worker count: an explicit [workerCount] is ignored here (it
/// is passed through by the caller); otherwise honour the `TSR_WORKERS`
/// environment variable if set, else default to the logical processor
/// count.
int _defaultWorkerCount() {
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

/// Worker entry point: announces itself, then processes jobs until it
/// receives the shutdown sentinel (null).
void _workerMain(SendPort sendPort) {
  final port = ReceivePort();
  sendPort.send(port.sendPort);
  port.listen((dynamic message) {
    if (message == null) {
      port.close();
      return;
    }
    final job = message as _ExtractionJob;
    DataVector? vector;
    try {
      final bytes = File(job.absolutePath).readAsBytesSync();
      final image = decodeImage(bytes);
      if (image == null) {
        stderr.writeln(
            'warning: failed to extract ${job.relativePath}: '
            'could not decode image');
      } else {
        vector = extractVector(job.relativePath, job.label, image);
      }
    } catch (e) {
      stderr.writeln('warning: failed to extract ${job.relativePath}: $e');
    }
    sendPort.send(<dynamic>[job.index, vector, port.sendPort]);
  });
}
