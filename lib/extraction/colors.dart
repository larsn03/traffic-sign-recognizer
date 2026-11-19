/// Color classification for traffic-sign pixels.
///
/// Mirrors the Go project's `colors.go`: an HSV-based classifier with hard
/// thresholds. Hue is in [0, 360); saturation and value are on a 0..255
/// scale. Achromatic colors (black/white) are resolved first; then red
/// (hue near 0/360), yellow, and blue — the first match wins.
library;

import 'dart:math' as math;

/// The categories a pixel (or averaged color) can fall into.
enum ColorCategory { red, yellow, blue, white, black, unknown }

/// Converts an 8-bit (or averaged) RGB triple to HSV.
///
/// Returns `(h, s, v)` with `h` in [0, 360) and `s`, `v` in 0..255.
(double, double, double) toHsv(num r, num g, num b) {
  final max = math.max(r, math.max(g, b));
  final min = math.min(r, math.min(g, b));
  final v = max.toDouble();
  final s = max == 0 ? 0.0 : 255.0 * (max - min) / max;
  var h = 0.0;
  if (max != min) {
    final d = max - min;
    if (max == r) {
      h = 60.0 * (((g - b) / d) % 6);
    } else if (max == g) {
      h = 60.0 * ((b - r) / d + 2);
    } else {
      h = 60.0 * ((r - g) / d + 4);
    }
    if (h < 0) h += 360;
  }
  return (h, s, v);
}

/// Classifies an RGB triple into a [ColorCategory].
ColorCategory isColor(num r, num g, num b) {
  final (h, s, v) = toHsv(r, g, b);
  if (v < 45) return ColorCategory.black;
  if (s < 45 && v > 150) return ColorCategory.white;
  if ((h >= 340 || h <= 20) && s > 50 && v > 50) {
    return ColorCategory.red;
  }
  if (h >= 28 && h <= 60 && s > 25 && v > 55) {
    return ColorCategory.yellow;
  }
  if (h >= 190 && h <= 250 && s > 45 && v > 45) {
    return ColorCategory.blue;
  }
  return ColorCategory.unknown;
}

/// A pixel counts as (near-)white when its minimum channel value is > 180.
///
/// This matches pure white and near-white backgrounds such as
/// (197, 197, 195) and is the predicate used for white-frame removal.
bool isWhitePixel(num r, num g, num b) =>
    math.min(r, math.min(g, b)) > 180;
