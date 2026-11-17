import 'dart:io';

import 'package:csv/csv.dart';

import '../data/data_set.dart';
import '../data/data_vector.dart';

/// Extracts feature data from a file (a CSV) into an in-memory [DataSet].
///
/// Each non-header row is interpreted as `label, path, <numeric features…>`,
/// mirroring the schema produced by the Go image extractor. The first row is
/// treated as the header and skipped, matching the Go `Traffic_Sign,Path`
/// layout. Using the same shape keeps the downstream KNN and decision-tree code
/// agnostic to whether vectors originated from images or from a prepared CSV.
class Extractor {
  /// Extracts every data row described by [pathToData] into a new [DataSet].
  DataSet extractToCsv(String pathToData) {
    final content = File(pathToData).readAsStringSync();
    return _extractFromContent(content);
  }

  /// Parses CSV [content] into a [DataSet], treating the first row as a header.
  static DataSet _extractFromContent(String content) {
    if (content.trim().isEmpty) {
      throw ArgumentError('CSV content is empty.');
    }

    final codec = CsvCodec(shouldParseNumbers: false, eol: '\n');
    var records = codec.decoder.convert(content);
    // Skip the header row, then drop blank rows that lack a label and path.
    if (records.length > 1) records = records.sublist(1);

    final data = <DataVector>[];
    for (final record in records) {
      if (record.length >= 2 &&
          !record.every((cell) => cell == null || cell.toString().trim().isEmpty)) {
        try {
          data.add(DataVector.fromNumericRecord(
            record.map((cell) => cell?.toString() ?? '').toList(),
          ));
        } on FormatException catch (e) {
          throw ArgumentError('Could not parse CSV row ${data.length + 1}: ${e.message}');
        }
      }
    }

    if (data.isEmpty) {
      throw ArgumentError('No data rows found in the provided content.');
    }

    return DataSet(data: data);
  }
}
