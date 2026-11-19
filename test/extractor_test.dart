import 'dart:io';

import 'package:image/image.dart' as img;
import 'package:test/test.dart';
import 'package:traffic_sign_recognizer/data/data_vector_set.dart';
import 'package:traffic_sign_recognizer/extraction/extractor.dart';

/// Draws a white 80x60 canvas with a red square that has a white
/// interior (a minimal "traffic sign"), written as PNG.
void writeSign(Directory dir, String name) {
  final image = img.Image(width: 80, height: 60);
  img.fill(image, color: image.getColor(255, 255, 255));
  for (var y = 10; y < 50; y++) {
    for (var x = 20; x < 60; x++) {
      image.setPixel(x, y, image.getColor(220, 30, 30));
    }
  }
  for (var y = 25; y < 35; y++) {
    for (var x = 35; x < 45; x++) {
      image.setPixel(x, y, image.getColor(255, 255, 255));
    }
  }
  File('${dir.path}/$name').writeAsBytesSync(img.encodePng(image));
}

void main() {
  late Directory tempDir;

  setUp(() {
    tempDir = Directory.systemTemp.createTempSync('extractor-test');
  });

  tearDown(() {
    tempDir.deleteSync(recursive: true);
  });

  test('extracts one vector per image with correct label', () async {
    final classDir = Directory('${tempDir.path}/Stop/0/80x60')
      ..createSync(recursive: true);
    writeSign(classDir, 'X0Y0.png');

    final set = await Extractor().extract(tempDir.path);
    expect(set.data.length, 1);
    final v = set.data.single;
    expect(v.trafficSign, 'Stop');
    expect(v.picturePath, 'Stop/0/80x60/X0Y0.png');
    expect(v.containsRed, isTrue);
    expect(v.containsBlue, isFalse);
    expect(v.containsYellow, isFalse);
    expect(v.containsBlack, isFalse);
    // Red + white are present.
    expect(v.numberOfDifferentColors, 2);
    // The red frame covers the corner cells; the white interior sits
    // inside the center cell, so it is mostly white.
    expect(v.whiteFractions[0], 0.0);
    expect(v.whiteFractions[4], greaterThan(0.5));
    expect(v.numberOfColorChangesVertically.length, 10);
    expect(v.numberOfColorChangesHorizontally.length, 10);
    expect(v.whiteFractions.length, 9);
    expect(v.leftRightAsymmetry.length, 9);
  });

  test('extractToCsv writes the 45-column schema', () async {
    final classDir = Directory('${tempDir.path}/Stop/0/80x60')
      ..createSync(recursive: true);
    writeSign(classDir, 'X0Y0.png');

    final csvPath = '${tempDir.path}/out.csv';
    await Extractor().extractToCsv(tempDir.path, csvPath);

    final content = File(csvPath).readAsStringSync();
    final lines = content.trim().split('\n');
    expect(lines.length, 2); // header + one row
    expect(lines[0].split(',').length, 45);
    expect(lines[1].split(',').length, 45);

    final set = DataVectorSet.fromCsv(csvPath);
    expect(set.data.single.trafficSign, 'Stop');
  });

  test('unsupported extensions are ignored', () async {
    final classDir = Directory('${tempDir.path}/Stop')
      ..createSync(recursive: true);
    File('${classDir.path}/notes.txt').writeAsStringSync('not an image');
    writeSign(classDir, 'X0Y0.png');

    final set = await Extractor().extract(tempDir.path);
    expect(set.data.length, 1);
  });
}
