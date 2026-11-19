import 'package:test/test.dart';
import 'package:traffic_sign_recognizer/extraction/colors.dart';

void main() {
  group('isColor', () {
    test('black: V < 45 wins over everything', () {
      expect(isColor(10, 10, 10), ColorCategory.black);
      expect(isColor(44, 44, 44), ColorCategory.black);
      // A very dark red is black, not red.
      expect(isColor(40, 10, 10), ColorCategory.black);
    });

    test('white: S < 45 && V > 150', () {
      expect(isColor(255, 255, 255), ColorCategory.white);
      expect(isColor(197, 197, 195), ColorCategory.white);
      // Mid grey: too bright for black, too dark for white → unknown.
      expect(isColor(100, 100, 100), ColorCategory.unknown);
      // Light but saturated is red, not white.
      expect(isColor(255, 200, 200), ColorCategory.red);
    });

    test('red: hue near 0/360', () {
      expect(isColor(230, 30, 30), ColorCategory.red);
      // Wraps around 360°.
      expect(isColor(200, 30, 80), ColorCategory.red);
    });

    test('yellow: hue 28..60', () {
      expect(isColor(255, 220, 0), ColorCategory.yellow);
    });

    test('blue: hue 190..250', () {
      expect(isColor(30, 60, 200), ColorCategory.blue);
    });

    test('unknown: outside all bands', () {
      // Green hue ~120 is not a sign color.
      expect(isColor(30, 180, 30), ColorCategory.unknown);
      // Hue 27.75 falls in the gap between the red (<=20) and yellow
      // (>=28) bands.
      expect(isColor(200, 157, 120), ColorCategory.unknown);
      // Saturated green is not black.
      expect(isColor(44, 200, 10), ColorCategory.unknown);
    });

    test('achromatic-first ordering', () {
      // Resolved before any hue band, even with saturation.
      expect(isColor(44, 44, 44), ColorCategory.black);
    });
  });

  group('isWhitePixel', () {
    test('min channel > 180', () {
      expect(isWhitePixel(255, 255, 255), isTrue);
      expect(isWhitePixel(197, 197, 195), isTrue);
      expect(isWhitePixel(180, 255, 255), isFalse);
      expect(isWhitePixel(181, 255, 255), isTrue);
    });
  });
}
