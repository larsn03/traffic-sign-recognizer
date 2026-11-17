import 'dart:async';
import 'dart:io';
import 'lib/extraction/image_extractor.dart';

FutureOr<void> main() async {
  final td = Directory('assets/training_data');
  final classes = td.listSync().whereType<Directory>().map((d) => d.path).toList();
  String? jpegClass;
  for (final c in classes) {
    if (Directory(c).listSync(recursive: true).any((e) => e is File && e.path.toLowerCase().endsWith('.jpg'))) {
      jpegClass = c;
      break;
    }
  }
  final jpeg = await ImageExtractor(maxImages: 3).extract(jpegClass!);
  print('JPEG rows: ${jpeg.length}');
  for (final v in jpeg.data.take(1)) {
    print('  colors=${v.numberOfDifferentColors} red=${v.containsRed}'
        ' black=${v.containsBlack} vert0=${v.numberOfColorChangesVertically.firstOrNull}'
        ' whiteFrac0=${v.whiteFractions.firstOrNull}');
  }

  final pngFile = ('assets/all_signs_png' as Directory)
      .listSync(recursive: true)
      .firstWhere((e) => e is File && e.path.toLowerCase().endsWith('.png'),
          orElse: () => throw Exception('no png'));
  final pngData = await ImageExtractor().extract(pngFile.path);
  print('PNG rows: ${pngData.length}');
  for (final v in pngData.data) {
    print('  colors=${v.numberOfDifferentColors} red=${v.containsRed}'
        ' black=${v.containsBlack} vert0=${v.numberOfColorChangesVertically.firstOrNull}'
        ' whiteFrac0=${v.whiteFractions.firstOrNull}');
  }

  final rows = jpeg.toCsvString().split('\n').where((l) => l.trim().isNotEmpty).toList();
  if (rows.length > 1) print('FULL JPEG ROW:   ' + rows[1]);
}

extension<T> on T? {
  T get firstOrNull => this as T;
}
