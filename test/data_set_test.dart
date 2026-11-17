import 'package:test/test.dart';
import 'package:traffic_sign_recognizer/data/data_set.dart';
import 'package:traffic_sign_recognizer/data/data_vector.dart';

void main() {
  test('loads and round-trips a feature CSV', () {
    final csv = '''Traffic_Sign,Path
Stop,photo1.jpg
4,2,0,0,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0.5,0.6,0.7,0.8,0.9,0.1,0.2,0.3,0.4''';
    final ds = DataSet.fromCsvString(csv);
    expect(ds.length, 2);
    expect(ds.classes, {'Stop', '4'});
    expect(ds.data.first.numericFeatures.length, DataVector.featureCount);
    final out = ds.toCsvString();
    expect(out.contains('Traffic_Sign,Path'), isTrue);
    expect(out.split('\n').length, 3);
  });
}
