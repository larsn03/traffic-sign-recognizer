import 'dart:io';

import 'package:test/test.dart';
import 'package:traffic_sign_recognizer/data/data_mask.dart';

void main() {
  group('parse', () {
    test('parses all six families', () {
      final mask = DataMask.parse('''
        number_of_different_colors: 2.0
        contains_colors: 0.5
        number_of_changes_vertically: 3
        number_of_changes_horizontally: 1.5
        white_fraction: 0
        left_right_asymmetry: 1
        ''');
      expect(mask.factors, [2.0, 0.5, 3.0, 1.5, 0.0, 1.0]);
    });

    test('ignores comments, blanks and unknown families', () {
      final mask = DataMask.parse('''
        # Factors for each family of parameters
        number_of_different_colors: 2.0

        some_other_thing: 9.9
        contains_colors: 0.5 # inline comment
        ''');
      expect(mask.factors, [2.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
    });

    test('missing families default to 1.0', () {
      expect(DataMask.parse('').factors,
          List<double>.filled(DataMask.familyCount, 1.0));
      expect(DataMask.parse('white_fraction: 0.25').factors,
          [1.0, 1.0, 1.0, 1.0, 0.25, 1.0]);
    });
  });

  group('apply', () {
    // Realistic vector length (43 features); exercises every boundary:
    // index 0 · 1..4 · 5..14 · 15..24 · 25..33 · 34..42.
    final features = List<double>.generate(43, (j) => j.toDouble());

    test('scales every feature by its family factor', () {
      final mask = DataMask([2, 3, 4, 5, 6, 7]);
      final out = mask.apply(features);
      for (var j = 0; j < features.length; j++) {
        final family = j == 0
            ? 0
            : j <= 4
                ? 1
                : j <= 14
                    ? 2
                    : j <= 24
                        ? 3
                        : j <= 33
                            ? 4
                            : 5;
        expect(out[j], features[j] * (family + 2), reason: 'index $j');
      }
    });

    test('identity mask leaves features unchanged', () {
      expect(DataMask.identity().apply(features), features);
    });

    test('does not modify the input', () {
      final input = List<double>.of(features);
      DataMask([2, 2, 2, 2, 2, 2]).apply(features);
      expect(features, input);
    });
  });

  group('save / loadOrCreate', () {
    test('round-trips through a file', () {
      final dir = Directory.systemTemp.createTempSync('tsr-mask-');
      try {
        final path = '${dir.path}/mask.yaml';
        final mask = DataMask([2, 0.5, 1, 1, 3, 1]);
        mask.save(path);
        expect(DataMask.parse(File(path).readAsStringSync()).factors,
            [2.0, 0.5, 1.0, 1.0, 3.0, 1.0]);
      } finally {
        dir.deleteSync(recursive: true);
      }
    });

    test('creates a default all-1.0 file when missing', () {
      final dir = Directory.systemTemp.createTempSync('tsr-mask-');
      try {
        final path = '${dir.path}/sub/mask.yaml'; // parent dirs created too
        final mask = DataMask.loadOrCreate(path);
        expect(mask.factors, List<double>.filled(DataMask.familyCount, 1.0));
        expect(File(path).existsSync(), isTrue);
        // A second load reads the written file.
        expect(DataMask.loadOrCreate(path).factors,
            List<double>.filled(DataMask.familyCount, 1.0));
      } finally {
        dir.deleteSync(recursive: true);
      }
    });

    test('loadOrCreate reads an existing file', () {
      final dir = Directory.systemTemp.createTempSync('tsr-mask-');
      try {
        final path = '${dir.path}/mask.yaml';
        File(path).writeAsStringSync('white_fraction: 0.1\n');
        expect(DataMask.loadOrCreate(path).factors,
            [1.0, 1.0, 1.0, 1.0, 0.1, 1.0]);
      } finally {
        dir.deleteSync(recursive: true);
      }
    });
  });
}
