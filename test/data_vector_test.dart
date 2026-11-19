import 'package:test/test.dart';
import 'package:traffic_sign_recognizer/data/data_vector.dart';
import 'package:traffic_sign_recognizer/data/data_vector_set.dart';

DataVector vector({
  String path = 'Stop/0/80x60/X0Y0.bmp',
  String sign = 'Stop',
  int colors = 2,
  bool red = true,
  bool blue = false,
  bool yellow = false,
  bool black = false,
  List<int> vert = const [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  List<int> hor = const [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
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
    expect(row[7], '0.1'); // vert[0] = 1/10, one decimal
    expect(row[16], '1.0'); // vert[9] = 10/10
    expect(row[17], '1.0'); // hor[0] = 10/10
    expect(row[26], '0.1'); // hor[9] = 1/10
    expect(row[27], '0.00'); // white[0], two decimals
    expect(row[31], '0.50'); // white[4]
    expect(row[36], '0.00'); // asym[0]
    expect(row[37], '-0.50'); // asym[1]
  });

  test('features() returns 43 numeric features with /10 scaling', () {
    final f = vector().features();
    expect(f.length, DataVector.featureCount);
    expect(f.length, 43);
    expect(f[0], 2.0);
    expect(f[1], 1.0);
    expect(f[2], 0.0);
    expect(f[5], 0.1); // vert[0]/10
    expect(f[14], 1.0); // vert[9]/10
    expect(f[15], 1.0); // hor[0]/10
    expect(f[24], 0.1); // hor[9]/10
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
}
