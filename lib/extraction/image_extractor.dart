import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img;

import '../data/data_set.dart';
import '../data/data_vector.dart';
import '../util/isolate_map.dart';

/// Internal colour categories used by [ImageExtractor], mirroring the Go
/// extractor's `isColor` (`data_extraction/colors.go`). Explicit ordering keeps
/// category ids stable.
enum _Color { red, yellow, blue, white, black }

const int _unknown = -1;

const Map<String, bool> _imageExtensions = {
  '.bmp': true,
  '.png': true,
  '.jpg': true,
  '.jpeg': true,
  '.gif': true,
  '.tif': true,
  '.tiff': true,
};

/// Decodes the images held under a directory (or a single file) and computes the
/// traffic-sign feature vector used by both classifiers.
///
/// Every image is analysed with the same routine the Go project uses: count
/// distinct colours and colour presence, strip the white frame around each sign,
/// then measure vertical/horizontal colour-change lines as well as per-cell
/// white fractions and left/right asymmetry — exactly filling the 43 numeric
/// columns described in [DataVector]. Keeping the feature layout shared means
/// KNN and the decision tree behave identically whether vectors came from images
/// or from a prepared CSV.
class ImageExtractor {
  /// Optional upper bound on the number of images decoded per class, handy for
  /// fast tests against a small subset of the assets. `null` means "all".
  final int? maxImages;

  const ImageExtractor({this.maxImages});

  /// Extracts features from [pathToData]: a directory (classes = subfolder
  /// names) or a single image/CSV file.
  Future<DataSet> extract(String pathToData) async {
    final file = File(pathToData);
    if (!file.existsSync() && !Directory(pathToData).existsSync()) {
      throw ArgumentError('No such path or directory: $pathToData');
    }

    if (file.existsSync()) {
      return _extractSingle(file.path);
    }

    // Collect (label, path) jobs up to maxImages before sharding the work, so
    // extraction can decode several images at once across isolates.
    final labels = <String>[];
    final paths = <String>[];
    var count = 0;
    for (final cls in _readClasses(pathToData)) {
      if (maxImages != null && count >= maxImages!) break;
      for (final image in _readImages('${pathToData}/$cls')) {
        labels.add(cls);
        paths.add(image);
        count++;
      }
    }

    final jobs = <(String, String)>[for (var i = 0; i < labels.length; i++) (labels[i], paths[i])];
    if (jobs.isEmpty) return DataSet(data: const []);

    // Shard decoding across isolates so independent images are extracted at once.
    final rows = (await isolateMap<(String, String)>(
      jobs,
      workerCount: _workersFor(jobs.length),
      produceChunk: (int start, int end) {
        final out = <Object?>[];
        for (var i = start; i < end; i++) {
          out.add(extractVector(jobs[i].$1, jobs[i].$2, jobs[i].$2));
        }
        return out;
      },
    )).whereType<DataVector>();

    return DataSet(data: rows.toList());
  }

  /// Upper bound on the isolates spawned for a batch of images. Each isolate
  /// decodes and features its own slice, so more workers means shorter runtimes;
  /// capped at eight to keep per-isolate memory bounded while still using several
  /// cores.
  int _workersFor(int count) => count <= 1 ? 1 : math.min(8, count);

  /// Extracts features from a single [path] whose label is taken from its file
  /// name without extension, or parses a CSV directly if one is supplied.
  DataSet _extractSingle(String path) {
    final lower = path.toLowerCase();
    if (lower.endsWith('.csv')) {
      return DataSet.fromCsvString(File(path).readAsStringSync());
    }

    final label = _labelFromPath(path);
    try {
      return DataSet(data: [extractVector(label, path, path)]);
    } on Object {
      return DataSet(data: <DataVector>[]);
    }
  }

  static String _labelFromPath(String path) {
    final name = path.split('/').last;
    return name.replaceAll(RegExp(r'\.[^./\\]+$'), '');
  }

  /// Sorted subfolder names that define the traffic-sign classes.
  List<String> _readClasses(String dir) => Directory(dir)
      .listSync()
      .whereType<Directory>()
      .map((d) => d.path.split('/').last)
      .toList()
    ..sort();

  /// Recursively gathers every image file within a class directory, sorted.
  List<String> _readImages(String dir) {
    final files = <String>[];
    for (final entry in Directory(dir).listSync(recursive: true)) {
      if (entry is File && _isImageFile(entry.path)) {
        files.add(entry.path);
      }
    }
    files.sort();
    return files;
  }

  static bool _isImageFile(String path) {
    final i = path.lastIndexOf('.');
    if (i == -1) return false;
    return _imageExtensions[path.substring(i).toLowerCase()] == true;
  }

  /// Builds a [DataVector] for one image using the Go `extractVector` routine.
  DataVector extractVector(String label, String absPath, String relPath) {
    final decoded = img.decodeImage(File(absPath).readAsBytesSync());
    if (decoded == null) {
      throw FormatException('Could not decode image $absPath');
    }

    final width = decoded.width;
    final height = decoded.height;

    // Cache r,g,b per pixel so every feature pass reads one source.
    final reds = <int>[];
    final greens = <int>[];
    final blues = <int>[];
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final px = decoded.getPixelSafe(x, y);
        reds.add(px.r.clamp(0, 255).round());
        greens.add(px.g.clamp(0, 255).round());
        blues.add(px.b.clamp(0, 255).round());
      }
    }

    final numColors = _analyzeColors(reds, greens, blues, width, height);
    var containsRed = false;
    var containsBlue = false;
    var containsYellow = false;
    var containsBlack = false;
    for (final c in _colorPresence(reds, greens, blues, width, height)) {
      switch (c) {
        case _Color.red:
          containsRed = true;
          break;
        case _Color.blue:
          containsBlue = true;
          break;
        case _Color.yellow:
          containsYellow = true;
          break;
        default: // black
          containsBlack = true;
      }
    }

    final (minX, minY, maxX, maxY) = _removeWhiteFrame(reds, greens, blues, width, height);
    final vert = _detectLines(reds, greens, blues, minX, minY, maxX, maxY, true, width);
    final horiz = _detectLines(reds, greens, blues, minX, minY, maxX, maxY, false, width);
    final whiteFractions = _computeWhiteFractions(reds, greens, blues, minX, minY, maxX, maxY, width);
    final asymmetry = _computeLeftRightAsymmetry(
      reds, greens, blues, minX, minY, maxX, maxY,
      width,
    );

    return DataVector(
      label: label,
      path: relPath,
      numberOfDifferentColors: numColors,
      containsRed: containsRed,
      containsBlue: containsBlue,
      containsYellow: containsYellow,
      containsBlack: containsBlack,
      numberOfColorChangesVertically: List<int>.of(vert),
      numberOfColorChangesHorizontally: List<int>.of(horiz),
      whiteFractions: <double>[...whiteFractions],
      leftRightAsymmetry: <double>[...asymmetry],
    );
  }

  /// Number of distinct colour categories present across the whole image.
  static int _analyzeColors(List<int> r, List<int> g, List<int> b, int w, int h) {
    return _colorPresence(r, g, b, w, h).length.clamp(0, 5);
  }

  /// Distinct colour categories found in the image, one per matching pixel.
  static Iterable<_Color> _colorPresence(List<int> r, List<int> g, List<int> b, int w, int h) {
    final seen = <_Color>{};
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final c = _classify(r[y * w + x], g[y * w + x], b[y * w + x]);
        if (c != null) seen.add(c);
      }
    }
    return seen;
  }

  static _Color? _classify(int r, int g, int b) {
    final (h, s, v) = _toHSV(
      double.parse(r.toString()),
      double.parse(g.toString()),
      double.parse(b.toString()),
    );

    if (v < 45) return _Color.black;
    if (s < 45 && v > 150) return _Color.white;
    if ((h < 20 || h > 340) && s > 50 && v > 50) return _Color.red;
    if (h >= 28 && h <= 60 && s > 25 && v > 55) return _Color.yellow;
    if (h >= 190 && h <= 250 && s > 45 && v > 45) return _Color.blue;
    return null; // unknown / not a recognised category
  }

  static const double _pi = math.pi;

  /// HSV triple, channels scaled to 0..255 for hue/sat and value.
  static (double, double, double) _toHSV(double r, double g, double b) {
    var max = r;
    if (g > max) max = g;
    if (b > max) max = b;
    var min = r;
    if (g < min) min = g;
    if (b < min) min = b;

    final value = max;
    final delta = max - min;

    double h = 0;
    double s;
    if (delta < 0.0001) {
      s = 0;
      h = 0;
    } else {
      s = max > 0 ? 255 * delta / max : 0;
      if (max == r) {
        h = (60 * ((g - b) / delta)) % 360;
        if (h < 0) h += 360;
      } else if (max == g) {
        h = 60 * ((b - r) / delta) + 120;
      } else {
        h = 60 * ((r - g) / delta) + 240;
      }
    }

    return (h, s, value);
  }

  static bool _isWhitePixel(int r, int g, int b) => math.min(r, math.min(g, b)) > 180;

  /// Bounding box of all non-white pixels (inclusive min + exclusive max).
  static (int, int, int, int) _removeWhiteFrame(
    List<int> r,
    List<int> g,
    List<int> b,
    int width,
    int height,
  ) {
    var minX = height;
    var minY = height;
    var maxX = -1;
    var maxY = -1;
    var found = false;
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final i = y * width + x;
        if (_isWhitePixel(r[i], g[i], b[i])) continue;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        found = true;
      }
    }
    return (found ? minX : 0, found ? minY : 0, found ? maxX + 1 : width, found ? maxY + 1 : height);
  }

  static List<int> _detectLines(
    List<int> r,
    List<int> g,
    List<int> b,
    int minX,
    int minY,
    int maxX,
    int maxY,
    bool vertical,
    int width,
  ) {
    final counts = <int>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    if (vertical) {
      final imageWidth = math.max(1, maxX - minX);
      if (imageWidth < 2) return counts;
      final lineLen = maxY - minY;
      final seg = math.max(1, lineLen ~/ 100);
      for (var i = 1; i <= 10; i++) {
        final x = minX + i * imageWidth ~/ 11;
        counts[i - 1] = _countLineChanges(r, g, b, x, minY, maxY, seg, true, width);
      }
    } else {
      final lineLen = math.max(1, maxX - minX);
      if (lineLen < 2) return counts;
      final imageHeight = math.max(1, maxY - minY);
      final seg = math.max(1, imageHeight ~/ 100);
      for (var i = 1; i <= 10; i++) {
        final y = minY + i * lineLen ~/ 11;
        counts[i - 1] = _countLineChanges(r, g, b, y, minX, maxX, seg, false, width);
      }
    }

    return counts;
  }

  static int _countLineChanges(
    List<int> r,
    List<int> g,
    List<int> b,
    int lineCoord,
    int start,
    int end,
    int seg,
    bool vertical,
    int width,
  ) {
    var changes = 0;
    Object? prevCat = _unknown;
    for (var pos = start; pos < end; pos += seg) {
      final endPos = math.min(pos + seg, end);
      var sumR = 0;
      var sumG = 0;
      var sumB = 0;
      var count = 0;
      for (var p = pos; p < endPos; p++) {
        final x = vertical ? lineCoord : p;
        final y = vertical ? p : lineCoord;
        final i = y * width + x;
        sumR += r[i];
        sumG += g[i];
        sumB += b[i];
        count++;
      }
      final avgR = count == 0 ? 0 : sumR ~/ count;
      final avgG = count == 0 ? 0 : sumG ~/ count;
      final avgB = count == 0 ? 0 : sumB ~/ count;
      final cat = _classify(avgR, avgG, avgB);
      if (cat != prevCat) {
        changes++;
        prevCat = cat;
      }
    }
    return changes;
  }

  static List<double> _computeWhiteFractions(
    List<int> r,
    List<int> g,
    List<int> b,
    int minX,
    int minY,
    int maxX,
    int maxY,
    int width,
  ) {
    final fractions = <double>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    final cellW = math.max(1, (maxX - minX) ~/ 3);
    final cellH = math.max(1, (maxY - minY) ~/ 3);
    for (var gy = 0; gy < 3; gy++) {
      for (var gx = 0; gx < 3; gx++) {
        final x0 = minX + gx * cellW;
        final x1 = math.min(x0 + cellW, maxX);
        final y0 = minY + gy * cellH;
        final y1 = math.min(y0 + cellH, maxY);
        var total = 0;
        var white = 0;
        for (var y = y0; y < y1; y++) {
          for (var x = x0; x < x1; x++) {
            final i = y * width + x;
            total++;
            if (_isWhitePixel(r[i], g[i], b[i])) white++;
          }
        }
        fractions[gy * 3 + gx] = total > 0 ? white / total : 0.0;
      }
    }
    return fractions;
  }

  static List<double> _computeLeftRightAsymmetry(
    List<int> r,
    List<int> g,
    List<int> b,
    int minX,
    int minY,
    int maxX,
    int maxY,
    int width,
  ) {
    final asym = <double>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    final cellW = math.max(1, (maxX - minX) ~/ 3);
    final cellH = math.max(1, (maxY - minY) ~/ 3);
    for (var gy = 0; gy < 3; gy++) {
      for (var gx = 0; gx < 3; gx++) {
        final x0 = minX + gx * cellW;
        final x1 = math.min(x0 + cellW, maxX);
        final y0 = minY + gy * cellH;
        final y1 = math.min(y0 + cellH, maxY);
        final midX = (x0 + x1) ~/ 2;
        var leftWhite = 0;
        var leftTotal = 0;
        var rightWhite = 0;
        var rightTotal = 0;
        for (var y = y0; y < y1; y++) {
          for (var x = x0; x < midX; x++) {
            final i = y * width + x;
            leftTotal++;
            if (_isWhitePixel(r[i], g[i], b[i])) leftWhite++;
          }
          for (var x = midX; x < x1; x++) {
            final i = y * width + x;
            rightTotal++;
            if (_isWhitePixel(r[i], g[i], b[i])) rightWhite++;
          }
        }
        final lf = leftTotal > 0 ? leftWhite / leftTotal : 0.0;
        final rf = rightTotal > 0 ? rightWhite / rightTotal : 0.0;
        asym[gy * 3 + gx] = lf - rf;
      }
    }
    return asym;
  }
}
