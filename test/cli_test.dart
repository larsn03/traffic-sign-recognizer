import 'dart:convert';
import 'dart:io';

import 'package:image/image.dart' as img;
import 'package:test/test.dart';

import '../bin/traffic_sign_recognizer.dart' as cli;

void main() {
  late Directory tempDir;

  setUp(() {
    tempDir = Directory.systemTemp.createTempSync('tsr-cli-test');
  });

  tearDown(() {
    tempDir.deleteSync(recursive: true);
  });

  /// Writes one solid-color PNG per class into the temp dir and extracts
  /// them to a CSV, so the training/classification modes have data to work
  /// on.
  Future<String> makeCsv(Map<String, (int, int, int)> colors) async {
    for (final entry in colors.entries) {
      final dir = Directory('${tempDir.path}/${entry.key}/0')
        ..createSync(recursive: true);
      final image = img.Image(width: 20, height: 20);
      img.fill(image,
          color: image.getColor(entry.value.$1, entry.value.$2, entry.value.$3));
      File('${dir.path}/X0Y0.png').writeAsBytesSync(img.encodePng(image));
    }
    final csv = '${tempDir.path}/data.csv';
    expect(await cli.run(['-e', tempDir.path, csv]), 0);
    return csv;
  }

  test('-dtree-train stores standardization parameters only with -z',
      () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });

    final rawTree = '${tempDir.path}/raw.json';
    expect(await cli.run(['-dtree-train', csv, rawTree]), 0);
    final rawJson =
        jsonDecode(File(rawTree).readAsStringSync()) as Map<String, dynamic>;
    expect(rawJson.containsKey('mean'), isFalse);
    expect(rawJson.containsKey('stddev'), isFalse);

    final zTree = '${tempDir.path}/z.json';
    expect(await cli.run(['-dtree-train', csv, zTree, '-z']), 0);
    final zJson =
        jsonDecode(File(zTree).readAsStringSync()) as Map<String, dynamic>;
    expect((zJson['mean'] as List).length, 43);
    expect((zJson['stddev'] as List).length, 43);
  });

  test('-dtree-classify requires -z to match the tree', () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });

    final zTree = '${tempDir.path}/z.json';
    expect(await cli.run(['-dtree-train', csv, zTree, '-z']), 0);
    final rawTree = '${tempDir.path}/raw.json';
    expect(await cli.run(['-dtree-train', csv, rawTree]), 0);

    final out = '${tempDir.path}/out.csv';
    // A z-trained tree demands -z ...
    expect(await cli.run(['-dtree-classify', zTree, csv, out]), 1);
    // ... and accepts -z.
    expect(await cli.run(['-dtree-classify', zTree, csv, out, '-z']), 0);
    // A raw tree rejects -z ...
    expect(await cli.run(['-dtree-classify', rawTree, csv, out, '-z']), 1);
    // ... and accepts no flag.
    expect(await cli.run(['-dtree-classify', rawTree, csv, out]), 0);
  });

  test('-dtree-train stores min-max parameters only with -minmax', () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });

    final rawTree = '${tempDir.path}/raw.json';
    expect(await cli.run(['-dtree-train', csv, rawTree]), 0);
    final rawJson =
        jsonDecode(File(rawTree).readAsStringSync()) as Map<String, dynamic>;
    expect(rawJson.containsKey('min'), isFalse);
    expect(rawJson.containsKey('max'), isFalse);

    final mmTree = '${tempDir.path}/mm.json';
    expect(await cli.run(['-dtree-train', csv, mmTree, '-minmax']), 0);
    final mmJson =
        jsonDecode(File(mmTree).readAsStringSync()) as Map<String, dynamic>;
    expect((mmJson['min'] as List).length, 43);
    expect((mmJson['max'] as List).length, 43);
  });

  test('standardization flags are mutually exclusive', () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });
    final tree = '${tempDir.path}/tree.json';
    expect(await cli.run(['-dtree-train', csv, tree]), 0);
    final out = '${tempDir.path}/out.csv';
    expect(await cli.run(['-dtree-train', csv, tree, '-z', '-minmax']), 1);
    expect(await cli.run(['-dtree-classify', tree, csv, out, '-z', '-minmax']),
        1);
    expect(await cli.run(['-knn', csv, csv, '2', '-z', '-minmax']), 1);
    expect(await cli.run(['-dtree-train', csv, tree, '-z', '-l1']), 1);
    expect(await cli.run(['-knn', csv, csv, '2', '-minmax', '-l1']), 1);
    expect(await cli.run(['-dtree-train', csv, tree, '-l1', '-l2']), 1);
    expect(await cli.run(['-knn', csv, csv, '2', '-z', '-l2']), 1);
  });

  test('-dtree-classify requires -minmax to match the tree', () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });

    final mmTree = '${tempDir.path}/mm.json';
    expect(await cli.run(['-dtree-train', csv, mmTree, '-minmax']), 0);
    final zTree = '${tempDir.path}/z.json';
    expect(await cli.run(['-dtree-train', csv, zTree, '-z']), 0);

    final out = '${tempDir.path}/out.csv';
    // A min-max-trained tree demands -minmax ...
    expect(await cli.run(['-dtree-classify', mmTree, csv, out]), 1);
    // ... rejects -z ...
    expect(await cli.run(['-dtree-classify', mmTree, csv, out, '-z']), 1);
    // ... and accepts -minmax.
    expect(await cli.run(['-dtree-classify', mmTree, csv, out, '-minmax']), 0);
    // A z-trained tree rejects -minmax.
    expect(await cli.run(['-dtree-classify', zTree, csv, out, '-minmax']), 1);
  });

  test('-knn runs with -minmax', () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });
    expect(await cli.run(['-knn', csv, csv, '2', '-minmax']), 0);
  });

  test('-dtree-train stores the l1 marker only with -l1', () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });

    final rawTree = '${tempDir.path}/raw.json';
    expect(await cli.run(['-dtree-train', csv, rawTree]), 0);
    final rawJson =
        jsonDecode(File(rawTree).readAsStringSync()) as Map<String, dynamic>;
    expect(rawJson.containsKey('standardization'), isFalse);

    final l1Tree = '${tempDir.path}/l1.json';
    expect(await cli.run(['-dtree-train', csv, l1Tree, '-l1']), 0);
    final l1Json =
        jsonDecode(File(l1Tree).readAsStringSync()) as Map<String, dynamic>;
    expect(l1Json['standardization'], 'l1');
    expect(l1Json.containsKey('mean'), isFalse);
    expect(l1Json.containsKey('min'), isFalse);
  });

  test('-dtree-classify requires -l1 to match the tree', () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });

    final l1Tree = '${tempDir.path}/l1.json';
    expect(await cli.run(['-dtree-train', csv, l1Tree, '-l1']), 0);
    final zTree = '${tempDir.path}/z.json';
    expect(await cli.run(['-dtree-train', csv, zTree, '-z']), 0);
    final rawTree = '${tempDir.path}/raw.json';
    expect(await cli.run(['-dtree-train', csv, rawTree]), 0);

    final out = '${tempDir.path}/out.csv';
    // An l1-trained tree demands -l1 ...
    expect(await cli.run(['-dtree-classify', l1Tree, csv, out]), 1);
    // ... rejects -z ...
    expect(await cli.run(['-dtree-classify', l1Tree, csv, out, '-z']), 1);
    // ... and accepts -l1.
    expect(await cli.run(['-dtree-classify', l1Tree, csv, out, '-l1']), 0);
    // A z-trained tree rejects -l1, and so does a raw tree.
    expect(await cli.run(['-dtree-classify', zTree, csv, out, '-l1']), 1);
    expect(await cli.run(['-dtree-classify', rawTree, csv, out, '-l1']), 1);
  });

  test('-knn runs with -l1', () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });
    expect(await cli.run(['-knn', csv, csv, '2', '-l1']), 0);
  });

  test('-dtree-train stores the l2 marker only with -l2', () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });

    final rawTree = '${tempDir.path}/raw.json';
    expect(await cli.run(['-dtree-train', csv, rawTree]), 0);
    final rawJson =
        jsonDecode(File(rawTree).readAsStringSync()) as Map<String, dynamic>;
    expect(rawJson.containsKey('standardization'), isFalse);

    final l2Tree = '${tempDir.path}/l2.json';
    expect(await cli.run(['-dtree-train', csv, l2Tree, '-l2']), 0);
    final l2Json =
        jsonDecode(File(l2Tree).readAsStringSync()) as Map<String, dynamic>;
    expect(l2Json['standardization'], 'l2');
    expect(l2Json.containsKey('mean'), isFalse);
    expect(l2Json.containsKey('min'), isFalse);
  });

  test('-dtree-classify requires -l2 to match the tree', () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });

    final l2Tree = '${tempDir.path}/l2.json';
    expect(await cli.run(['-dtree-train', csv, l2Tree, '-l2']), 0);
    final l1Tree = '${tempDir.path}/l1.json';
    expect(await cli.run(['-dtree-train', csv, l1Tree, '-l1']), 0);
    final rawTree = '${tempDir.path}/raw.json';
    expect(await cli.run(['-dtree-train', csv, rawTree]), 0);

    final out = '${tempDir.path}/out.csv';
    // An l2-trained tree demands -l2 ...
    expect(await cli.run(['-dtree-classify', l2Tree, csv, out]), 1);
    // ... rejects -l1 ...
    expect(await cli.run(['-dtree-classify', l2Tree, csv, out, '-l1']), 1);
    // ... and accepts -l2.
    expect(await cli.run(['-dtree-classify', l2Tree, csv, out, '-l2']), 0);
    // An l1-trained tree rejects -l2, and so does a raw tree.
    expect(await cli.run(['-dtree-classify', l1Tree, csv, out, '-l2']), 1);
    expect(await cli.run(['-dtree-classify', rawTree, csv, out, '-l2']), 1);
  });

  test('-knn runs with -l2', () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });
    expect(await cli.run(['-knn', csv, csv, '2', '-l2']), 0);
  });

  test('-knn runs with and without -z', () async {
    final csv = await makeCsv({
      'A': (220, 30, 30),
      'B': (30, 30, 220),
    });
    // k=2: the test set has only two training samples.
    expect(await cli.run(['-knn', csv, csv, '2']), 0);
    expect(await cli.run(['-knn', csv, csv, '2', '-z']), 0);
  });
}
