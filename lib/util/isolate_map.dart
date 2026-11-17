import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;

/// A single chunk of results returned by a worker isolate.
class _Chunk {
  final int index;
  final Object? rows;

  _Chunk(this.index, this.rows);
}

/// A chunk worker: given the contiguous range `[start, end)` it returns the
/// results it computed for that slice. Synchronous workers may simply return a
/// list; the caller awaits nothing.
typedef ChunkFunction = FutureOr<List<Object?>> Function(int start, int end);

/// Runs [produceChunk] over every contiguous slice of [items], distributing the
/// work across up to [workerCount] isolates so independent units are processed at
/// once, and returns the results concatenated in input order.
///
/// The default worker count is chosen from the number of items (a handful for
/// small sets, capped so a giant batch doesn't spawn an explosion of isolates).
Future<List<Object?>> isolateMap<T>(
  List<T> items, {
  int? workerCount,
  required ChunkFunction produceChunk,
}) async {
  final total = items.length;
  final workers = math.max(1, _resolveWorkers(total, workerCount));

  if (Platform.environment['ISOLATE_DEBUG'] != null) {
    print('isolateMap: sharding $total items across $workers isolates');
  }

  if (total == 0) return <Object?>[];
  if (workers == 1) return produceChunk(0, total);

  final out = <Object?>[for (int i = 0; i < workers; i++) null];
  var received = 0;
  final port = ReceivePort();
  for (var i = 0; i < workers; i++) {
    final start = i * total ~/ workers;
    final end = (i + 1) * total ~/ workers;
    Isolate.spawn((SendPort send) => send.send(_Chunk(i, produceChunk(start, end))), port.sendPort);
  }
  await for (final msg in port) {
    final chunk = msg as _Chunk;
    out[chunk.index] = chunk.rows;
    if (++received == workers) break;
  }
  port.close();

  final result = <Object?>[];
  for (final rows in out) {
    if (rows != null) result.add(rows as List<Object?>);
  }
  return result;
}

/// Picks a sane isolate count without needing [dart:io]'s `Platform`.
int _resolveWorkers(int total, int? requested) {
  final base = requested ?? _defaultWorkerCount;
  if (total <= 1) return 1;
  return math.min(base, total);
}

/// A reasonable default isolate count for CPU-bound batches. The heavy decode +
/// feature pass per image is memory-light (each isolate holds one or two images'
/// worth of buffers), so a handful of workers buys real parallelism here.
const int _defaultWorkerCount = 8;
