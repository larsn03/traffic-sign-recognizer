import 'lib/extraction/image_extractor.dart';
import 'lib/data/data_set.dart';

void main() {
  final ext = ImageExtractor(maxImages: 3);
  final ds = ext.extract('assets/to_be_classified_data/Stop');
  print('Rows extracted: ${ds.length}');
  for (final v in ds.data) {
    print(
      'label=${v.label} path=${v.path.split("/").last}'
      ' colors=${v.numberOfDifferentColors}'
      ' red=${v.containsRed} yellow=${v.containsYellow}'
      ' blue=${v.containsBlue} black=${v.containsBlack}'
      ' vert0=${v.numberOfColorChangesVertically.firstOrNull} '
      'vert1=${v.numberOfColorChangesVertically[1].orZero} '
      'whiteFrac0=${v.whiteFractions.firstOrNull}',
    );
  }
  print('--- header + first row ---');
  final lines = ds.toCsvString().split('\n').where((l) => l.trim().isNotEmpty).toList();
  for (var i = 0; i < 2 && i < lines.length; i++) print(lines[i]);
}

extension<T> on T? {
  T get orZero => this as T;
  int get orThree => (this as num?).toInt() ?? -1;
}
