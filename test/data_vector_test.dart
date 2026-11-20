import 'dart:math' as math;

import 'package:test/test.dart';
import 'package:traffic_sign_recognizer/data/data_vector.dart';
import 'package:traffic_sign_recognizer/data/data_vector_set.dart';

DataVector vector({
  String path = 'Stop/0/80x60/X0Y0.bmp',
  String sign = 'Stop',
  double colors = 2,
  double red = 1,
  double blue = 0,
  double yellow = 0,
  double black = 0,
  List<double> vert = const [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  List<double> hor = const [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
  List<double> white = const [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0],
  List<double> asym = const [0.0, -0.5, 0.25, 0.33, 0.0, -0.33, 0.5, -0.25, 0.0],
}) {
  return DataVector(
    picturePath: path,
    trafficSign: sign,
    numberOfDifferentColors: colors,
    containsRed: red,
    containsBlue: blue,
    containsYellow: yellow,
    containsBlack: black,
    numberOfColorChangesVertically: vert,
    numberOfColorChangesHorizontally: hor,
    whiteFractions: white,
    leftRightAsymmetry: asym,
  );
}

void main() {
  test('toRow produces 45 fields in canonical order', () {
    final row = vector().toRow();
    expect(row.length, 45);
    expect(row[0], 'Stop');
    expect(row[1], 'Stop/0/80x60/X0Y0.bmp');
    expect(row[2], '2');
    expect(row[3], '1'); // red
    expect(row[4], '0'); // blue
    expect(row[5], '0'); // yellow
    expect(row[6], '0'); // black
    expect(row[7], '1'); // vert[0] = 1, raw count
    expect(row[16], '10'); // vert[9] = 10, raw count
    expect(row[17], '10'); // hor[0] = 10, raw count
    expect(row[26], '1'); // hor[9] = 1, raw count
    expect(row[27], '0.00'); // white[0], two decimals
    expect(row[31], '0.50'); // white[4]
    expect(row[36], '0.00'); // asym[0]
    expect(row[37], '-0.50'); // asym[1]
  });

  test('features returns 43 raw numeric features', () {
    final f = vector().features;
    expect(f.length, DataVector.featureCount);
    expect(f.length, 43);
    expect(f[0], 2.0);
    expect(f[1], 1.0);
    expect(f[2], 0.0);
    expect(f[5], 1.0); // vert[0]
    expect(f[14], 10.0); // vert[9]
    expect(f[15], 10.0); // hor[0]
    expect(f[24], 1.0); // hor[9]
    expect(f[29], 0.5); // white[4]
    expect(f[35], -0.5); // asym[1]
    expect(f[42], 0.0); // asym[8]
  });

  test('fromRow round-trips toRow', () {
    final original = vector();
    final decoded = DataVector.fromRow(original.toRow());
    expect(decoded.trafficSign, original.trafficSign);
    expect(decoded.picturePath, original.picturePath);
    expect(decoded.numberOfDifferentColors, original.numberOfDifferentColors);
    expect(decoded.containsRed, original.containsRed);
    expect(decoded.containsBlue, original.containsBlue);
    expect(decoded.containsYellow, original.containsYellow);
    expect(decoded.containsBlack, original.containsBlack);
    expect(decoded.numberOfColorChangesVertically,
        original.numberOfColorChangesVertically);
    expect(decoded.numberOfColorChangesHorizontally,
        original.numberOfColorChangesHorizontally);
    expect(decoded.whiteFractions, original.whiteFractions);
    expect(decoded.leftRightAsymmetry, original.leftRightAsymmetry);
  });

  test('DataVectorSet header has 45 column names', () {
    expect(DataVectorSet.header.length, 45);
    expect(DataVectorSet.header.first, 'Traffic_Sign');
    expect(DataVectorSet.header[7], 'Number_of_color_changes_vertically_0');
    expect(DataVectorSet.header[17], 'Number_of_color_changes_horizontally_0');
    expect(DataVectorSet.header[27], 'White_fraction_0');
    expect(DataVectorSet.header[36], 'Left_right_asymmetry_0');
    expect(DataVectorSet.header.last, 'Left_right_asymmetry_8');
  });

  test('standardizeZ fits on this set and rewrites features in place', () {
    final a = vector();
    final b = vector(colors: 4);
    final set = DataVectorSet(data: [a, b]);
    final (mean, stddev) = set.standardizeZ();

    // Only the color-count column varies: mean 3, population stddev 1.
    expect(mean.length, DataVector.featureCount);
    expect(mean[0], 3.0);
    expect(stddev[0], 1.0);
    expect(a.features[0], -1.0);
    expect(b.features[0], 1.0);
    // Zero-variance columns map to 0.
    for (var j = 1; j < DataVector.featureCount; j++) {
      expect(stddev[j], 0);
      expect(a.features[j], 0);
      expect(b.features[j], 0);
    }
    // The raw typed fields — and thus the CSV encoding — are untouched.
    expect(a.numberOfDifferentColors, 2.0);
    expect(b.numberOfDifferentColors, 4.0);
    expect(a.toRow(), vector().toRow());
  });

  test('standardizeZ applies externally fitted parameters', () {
    final train = DataVectorSet(data: [vector(), vector(colors: 4)]);
    final params = train.standardizeZ();
    final testSet = DataVectorSet(data: [vector()]);
    final applied =
        testSet.standardizeZ(mean: params.$1, stddev: params.$2);
    expect(applied.$1, same(params.$1));
    expect(applied.$2, same(params.$2));

    final f = testSet.data.single.features;
    // One below the training mean in the varying column...
    expect(f[0], closeTo(-1.0, 1e-12));
    // ...and 0 in the zero-variance columns.
    for (var j = 1; j < DataVector.featureCount; j++) {
      expect(f[j], 0);
    }
  });

  test('standardizeZ on an empty set returns zero parameters', () {
    final (mean, stddev) = DataVectorSet().standardizeZ();
    expect(mean.length, DataVector.featureCount);
    expect(stddev.length, DataVector.featureCount);
    expect(mean, everyElement(0));
    expect(stddev, everyElement(0));
  });

  test('minMaxStandardize fits on this set and rewrites features in place',
      () {
    final a = vector();
    final b = vector(colors: 4);
    final set = DataVectorSet(data: [a, b]);
    final (min, max) = set.minMaxStandardize();

    // Only the color-count column varies: it spans [2, 4].
    expect(min.length, DataVector.featureCount);
    expect(min[0], 2.0);
    expect(max[0], 4.0);
    expect(a.features[0], 0.0);
    expect(b.features[0], 1.0);
    // Zero-range columns map to 0.
    for (var j = 1; j < DataVector.featureCount; j++) {
      expect(a.features[j], 0);
      expect(b.features[j], 0);
    }
    // The raw typed fields — and thus the CSV encoding — are untouched.
    expect(a.numberOfDifferentColors, 2.0);
    expect(b.numberOfDifferentColors, 4.0);
    expect(a.toRow(), vector().toRow());
  });

  test('minMaxStandardize applies externally fitted parameters', () {
    final train = DataVectorSet(data: [vector(), vector(colors: 4)]);
    final params = train.minMaxStandardize();
    final testSet = DataVectorSet(data: [vector()]);
    final applied =
        testSet.minMaxStandardize(min: params.$1, max: params.$2);
    expect(applied.$1, same(params.$1));
    expect(applied.$2, same(params.$2));

    final f = testSet.data.single.features;
    // The test vector equals the training minimum in the varying column
    // ... and 0 in the zero-range columns.
    expect(f[0], 0.0);
    for (var j = 1; j < DataVector.featureCount; j++) {
      expect(f[j], 0);
    }
  });

  test('minMaxStandardize on an empty set returns zero parameters', () {
    final (min, max) = DataVectorSet().minMaxStandardize();
    expect(min.length, DataVector.featureCount);
    expect(max.length, DataVector.featureCount);
    expect(min, everyElement(0));
    expect(max, everyElement(0));
  });

  test('robustStandardize fits on this set and rewrites features in place',
      () {
    final a = vector();
    final b = vector(colors: 4);
    final set = DataVectorSet(data: [a, b]);
    final (median, iqr) = set.robustStandardize();

    // Only the color-count column varies: median 3, IQR 1 (Q3 3.5, Q1 2.5).
    expect(median.length, DataVector.featureCount);
    expect(median[0], 3.0);
    expect(iqr[0], 1.0);
    expect(a.features[0], -1.0);
    expect(b.features[0], 1.0);
    // Zero-IQR columns map to 0.
    for (var j = 1; j < DataVector.featureCount; j++) {
      expect(iqr[j], 0);
      expect(a.features[j], 0);
      expect(b.features[j], 0);
    }
    // The raw typed fields — and thus the CSV encoding — are untouched.
    expect(a.numberOfDifferentColors, 2.0);
    expect(b.numberOfDifferentColors, 4.0);
    expect(a.toRow(), vector().toRow());
  });

  test('robustStandardize fits quantiles by linear interpolation', () {
    final set = DataVectorSet(data: [
      vector(colors: 1),
      vector(colors: 2),
      vector(colors: 3),
      vector(colors: 4),
    ]);
    final (median, iqr) = set.robustStandardize();

    // Column [1, 2, 3, 4]: median 2.5, Q1 1.75, Q3 3.25, IQR 1.5.
    expect(median[0], 2.5);
    expect(iqr[0], 1.5);
    expect(set.data[0].features[0], -1.0);
    expect(set.data[1].features[0], closeTo(-1 / 3, 1e-12));
    expect(set.data[2].features[0], closeTo(1 / 3, 1e-12));
    expect(set.data[3].features[0], 1.0);
  });

  test('robustStandardize applies externally fitted parameters', () {
    final train = DataVectorSet(data: [vector(), vector(colors: 4)]);
    final params = train.robustStandardize();
    final testSet = DataVectorSet(data: [vector()]);
    final applied =
        testSet.robustStandardize(median: params.$1, iqr: params.$2);
    expect(applied.$1, same(params.$1));
    expect(applied.$2, same(params.$2));

    final f = testSet.data.single.features;
    // One below the training median in the varying column...
    expect(f[0], closeTo(-1.0, 1e-12));
    // ...and 0 in the zero-IQR columns.
    for (var j = 1; j < DataVector.featureCount; j++) {
      expect(f[j], 0);
    }
  });

  test('robustStandardize on an empty set returns zero parameters', () {
    final (median, iqr) = DataVectorSet().robustStandardize();
    expect(median.length, DataVector.featureCount);
    expect(iqr.length, DataVector.featureCount);
    expect(median, everyElement(0));
    expect(iqr, everyElement(0));
  });

  test('l1Standardize divides the vector by its L1 norm', () {
    final v = vector();
    v.l1Standardize();
    // L1 norm of the default vector: 2 + 1 + 55 + 55 + 4.2 + 2.16 = 119.36
    // (the raw color-change counts sum to 55 each; the asymmetry features
    // contribute their absolute values).
    expect(v.features[0], closeTo(2 / 119.36, 1e-12));
    expect(v.features[1], closeTo(1 / 119.36, 1e-12));
    expect(
        v.features.fold(0.0, (sum, f) => sum + f.abs()), closeTo(1.0, 1e-9));
    // The raw typed fields — and thus the CSV encoding — are untouched.
    expect(v.numberOfDifferentColors, 2.0);
    expect(v.toRow(), vector().toRow());
  });

  test('l1Standardize maps a single non-zero feature to 1', () {
    final v = DataVector(
      picturePath: 'p',
      trafficSign: 'A',
      numberOfDifferentColors: 2,
      containsRed: 0,
      containsBlue: 0,
      containsYellow: 0,
      containsBlack: 0,
      numberOfColorChangesVertically: List<double>.filled(10, 0),
      numberOfColorChangesHorizontally: List<double>.filled(10, 0),
      whiteFractions: List<double>.filled(9, 0),
      leftRightAsymmetry: List<double>.filled(9, 0),
    );
    v.l1Standardize();
    expect(v.features[0], 1.0);
    for (var j = 1; j < DataVector.featureCount; j++) {
      expect(v.features[j], 0);
    }
  });

  test('l1Standardize leaves a zero-norm vector untouched', () {
    final v = DataVector(
      picturePath: 'p',
      trafficSign: 'A',
      numberOfDifferentColors: 0,
      containsRed: 0,
      containsBlue: 0,
      containsYellow: 0,
      containsBlack: 0,
      numberOfColorChangesVertically: List<double>.filled(10, 0),
      numberOfColorChangesHorizontally: List<double>.filled(10, 0),
      whiteFractions: List<double>.filled(9, 0),
      leftRightAsymmetry: List<double>.filled(9, 0),
    );
    v.l1Standardize();
    expect(v.features, everyElement(0));
  });

  test('l1Standardize on a set transforms every vector', () {
    final set = DataVectorSet(data: [vector(), vector(colors: 4)]);
    set.l1Standardize();
    for (final v in set.data) {
      expect(v.features.fold(0.0, (sum, f) => sum + f.abs()),
          closeTo(1.0, 1e-9));
    }
  });

  test('l2Standardize divides the vector by its L2 norm', () {
    final v = vector();
    final original = List<double>.of(v.features);
    v.l2Standardize();
    final norm = math.sqrt(original.fold(0.0, (sum, f) => sum + f * f));
    for (var j = 0; j < DataVector.featureCount; j++) {
      expect(v.features[j], closeTo(original[j] / norm, 1e-12));
    }
    expect(v.features.fold(0.0, (sum, f) => sum + f * f), closeTo(1.0, 1e-9));
    // The raw typed fields — and thus the CSV encoding — are untouched.
    expect(v.numberOfDifferentColors, 2.0);
    expect(v.toRow(), vector().toRow());
  });

  test('l2Standardize maps a single non-zero feature to 1', () {
    final v = DataVector(
      picturePath: 'p',
      trafficSign: 'A',
      numberOfDifferentColors: 2,
      containsRed: 0,
      containsBlue: 0,
      containsYellow: 0,
      containsBlack: 0,
      numberOfColorChangesVertically: List<double>.filled(10, 0),
      numberOfColorChangesHorizontally: List<double>.filled(10, 0),
      whiteFractions: List<double>.filled(9, 0),
      leftRightAsymmetry: List<double>.filled(9, 0),
    );
    v.l2Standardize();
    expect(v.features[0], 1.0);
    for (var j = 1; j < DataVector.featureCount; j++) {
      expect(v.features[j], 0);
    }
  });

  test('l2Standardize leaves a zero-norm vector untouched', () {
    final v = DataVector(
      picturePath: 'p',
      trafficSign: 'A',
      numberOfDifferentColors: 0,
      containsRed: 0,
      containsBlue: 0,
      containsYellow: 0,
      containsBlack: 0,
      numberOfColorChangesVertically: List<double>.filled(10, 0),
      numberOfColorChangesHorizontally: List<double>.filled(10, 0),
      whiteFractions: List<double>.filled(9, 0),
      leftRightAsymmetry: List<double>.filled(9, 0),
    );
    v.l2Standardize();
    expect(v.features, everyElement(0));
  });

  test('l2Standardize on a set transforms every vector', () {
    final set = DataVectorSet(data: [vector(), vector(colors: 4)]);
    set.l2Standardize();
    for (final v in set.data) {
      expect(v.features.fold(0.0, (sum, f) => sum + f * f), closeTo(1.0, 1e-9));
    }
  });
}
