import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart';

import '../data/data_vector.dart';
import 'colors.dart';

/// Extracts the feature vector for one decoded image.
///
/// Mirrors the Go project's extraction pipeline:
///  1. color analysis over the full image,
///  2. white-frame removal (bounding box of non-white pixels),
///  3. vertical color-change detection (10 lines),
///  4. horizontal color-change detection (10 lines),
///  5. white fractions over a 3x3 grid,
///  6. left-right asymmetry over the same grid.
DataVector extractVector(String picturePath, String trafficSign, Image img) {
  final width = img.width;
  final height = img.height;

  // Obtain a tightly packed byte buffer plus the bytes-per-pixel stride. For
  // 8-bit images without row padding we use the native buffer directly and
  // simply ignore the alpha channel (the colour analysis never reads it);
  // anything else is normalised to RGBA8.
  late final Uint8List buffer;
  late final int stride;
  if (img.format == Format.uint8 &&
      (img.numChannels == 3 || img.numChannels == 4) &&
      img.rowStride == width * img.numChannels) {
    buffer = img.getBytes();
    stride = img.numChannels;
  } else {
    buffer = toRgba8(img);
    stride = 4;
  }

  // 1. Color analysis (full image) + 2. white-frame bounding box, in one pass.
  final present = <ColorCategory>{};
  var minX = width, minY = height, maxX = -1, maxY = -1;
  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      final i = (y * width + x) * stride;
      final r = buffer[i], g = buffer[i + 1], b = buffer[i + 2];
      present.add(isColor(r, g, b));
      if (!isWhitePixel(r, g, b)) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  present.remove(ColorCategory.unknown);

  // Fall back to the full image bounds so subsequent detectors return
  // deterministic zeros when the image is entirely white.
  if (maxX < 0) {
    minX = 0;
    minY = 0;
    maxX = width - 1;
    maxY = height - 1;
  }
  final cropW = maxX - minX + 1;
  final cropH = maxY - minY + 1;

  // 3. + 4. Color-change detection: 10 vertical and 10 horizontal lines at
  // 1/11 ... 10/11 of the cropped size.
  final verticalChanges = <int>[];
  for (var i = 1; i <= 10; i++) {
    final x = minX + (cropW * i) ~/ 11;
    verticalChanges.add(_countLineChanges(
        buffer, width, stride, x, minX, maxX, minY, maxY, true));
  }
  final horizontalChanges = <int>[];
  for (var i = 1; i <= 10; i++) {
    final y = minY + (cropH * i) ~/ 11;
    horizontalChanges.add(_countLineChanges(
        buffer, width, stride, y, minX, maxX, minY, maxY, false));
  }

  // 5. + 6. White fractions and left-right asymmetry over a 3x3 grid.
  final whiteFractions = List<double>.filled(9, 0);
  final leftRightAsymmetry = List<double>.filled(9, 0);
  for (var row = 0; row < 3; row++) {
    final y0 = minY + (cropH * row) ~/ 3;
    final y1 = minY + (cropH * (row + 1)) ~/ 3;
    for (var col = 0; col < 3; col++) {
      final x0 = minX + (cropW * col) ~/ 3;
      final x1 = minX + (cropW * (col + 1)) ~/ 3;
      final midX = (x0 + x1) ~/ 2;
      var white = 0, total = 0;
      var leftWhite = 0, leftTotal = 0, rightWhite = 0, rightTotal = 0;
      for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) {
          final i = (y * width + x) * stride;
          final whiteHere = isWhitePixel(buffer[i], buffer[i + 1], buffer[i + 2]);
          total++;
          if (whiteHere) white++;
          if (x < midX) {
            leftTotal++;
            if (whiteHere) leftWhite++;
          } else {
            rightTotal++;
            if (whiteHere) rightWhite++;
          }
        }
      }
      final cell = row * 3 + col;
      whiteFractions[cell] = total == 0 ? 0.0 : white / total;
      final leftFraction = leftTotal == 0 ? 0.0 : leftWhite / leftTotal;
      final rightFraction = rightTotal == 0 ? 0.0 : rightWhite / rightTotal;
      leftRightAsymmetry[cell] = leftFraction - rightFraction;
    }
  }

  return DataVector(
    picturePath: picturePath,
    trafficSign: trafficSign,
    numberOfDifferentColors: present.length,
    containsRed: present.contains(ColorCategory.red),
    containsBlue: present.contains(ColorCategory.blue),
    containsYellow: present.contains(ColorCategory.yellow),
    containsBlack: present.contains(ColorCategory.black),
    numberOfColorChangesVertically: verticalChanges,
    numberOfColorChangesHorizontally: horizontalChanges,
    whiteFractions: whiteFractions,
    leftRightAsymmetry: leftRightAsymmetry,
  );
}

/// Normalizes any decoded image to tightly packed RGBA8 (4 bytes per
/// pixel), mirroring the Go project's `imageToRGBA` step. Images without
/// an alpha channel are treated as opaque; higher bit depths are scaled
/// down to 8 bits.
Uint8List toRgba8(Image img) {
  final width = img.width;
  final height = img.height;
  final rgba = Uint8List(width * height * 4);
  final hasAlpha = img.numChannels == 4;
  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      final c = img.getPixel(x, y);
      final i = (y * width + x) * 4;
      rgba[i] = (c.rNormalized * 255).round();
      rgba[i + 1] = (c.gNormalized * 255).round();
      rgba[i + 2] = (c.bNormalized * 255).round();
      rgba[i + 3] = hasAlpha ? (c.aNormalized * 255).round() : 255;
    }
  }
  return rgba;
}

/// Scans one line of the cropped image top-to-bottom (vertical) or
/// left-to-right (horizontal).
///
/// The line is sampled in segments of `seg = lineLen / 100` pixels (floored
/// to 1); each segment's average RGB is classified into a single category,
/// and the number of category transitions between consecutive segments is
/// returned. White-to-white is not a change because the category stays the
/// same; transitions to/from `unknown` do count.
int _countLineChanges(
  Uint8List buffer,
  int width,
  int stride,
  int fixedCoordinate,
  int minX,
  int maxX,
  int minY,
  int maxY,
  bool vertical,
) {
  final lineLen = vertical ? maxY - minY + 1 : maxX - minX + 1;
  final seg = math.max(1, lineLen ~/ 100);
  ColorCategory? previous;
  var changes = 0;
  for (var start = 0; start < lineLen; start += seg) {
    final end = math.min(start + seg, lineLen);
    var r = 0.0, g = 0.0, b = 0.0;
    for (var offset = start; offset < end; offset++) {
      final x = vertical ? fixedCoordinate : minX + offset;
      final y = vertical ? minY + offset : fixedCoordinate;
      final i = (y * width + x) * stride;
      r += buffer[i];
      g += buffer[i + 1];
      b += buffer[i + 2];
    }
    final n = end - start;
    final category = isColor(r / n, g / n, b / n);
    if (previous != null && category != previous) changes++;
    previous = category;
  }
  return changes;
}
