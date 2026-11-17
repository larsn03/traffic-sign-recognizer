import 'dart:io';

import 'package:csv/csv.dart';

import 'data_vector.dart';

/// A collection of [DataVector]s that can be classified.
///
/// [DataSet] is the in-memory model shared by extractors, learners and
/// classifiers. It deliberately keeps its storage simple (an exposed mutable
/// list) so both high-level flows and lower level callers stay flexible.
class DataSet {
  /// The raw feature vectors held by this set. Mutate with care.
  List<DataVector> data;

  DataSet({this.data = const []});

  /// The number of vectors held by this set.
  int get length => data.length;

  /// Whether the set currently holds no vectors.
  bool get isEmpty => data.isEmpty;

  /// Whether the set currently holds at least one vector.
  bool get isNotEmpty => data.isNotEmpty;

  /// Adds a single [vector] to the set, returning the updated set for chaining.
  DataSet add(DataVector vector) {
    data.add(vector);
    return this;
  }

  /// Returns all distinct labels (class names) present in this set.
  Set<String> get classes => data.map((v) => v.label).toSet();

  /// Loads the feature vectors described by [path] into a new [DataSet].
  ///
  /// Each CSV row is interpreted as `label, path, <numeric features…>` which is
  /// exactly the schema produced by the Go extractor and consumed by KNN.
  static DataSet fromCsv(String path) {
    final content = File(path).readAsStringSync();
    return DataSet.fromCsvString(content);
  }

  /// Parses CSV [content] into a new [DataSet].
  static DataSet fromCsvString(String content) {
    if (content.trim().isEmpty) {
      throw ArgumentError('CSV content is empty.');
    }

    final codec = CsvCodec(
      shouldParseNumbers: false,
      eol: '\n',
      fieldDelimiter: ',',
    );
    final rows = codec.decoder.convert(content);
    if (rows.isEmpty) {
      throw ArgumentError('CSV has no data rows to load.');
    }

    // The first row is the header (`Traffic_Sign,Path…`); treat it as such.
    var parsed = rows.length > 1 ? rows.sublist(1) : rows;

    final data = <DataVector>[];
    for (final row in parsed) {
      if (row.length >= 2 && !row.every((c) => c == null || c.toString().trim().isEmpty)) {
        try {
          data.add(DataVector.fromNumericRecord(row.map((e) => e?.toString() ?? '').toList()));
        } on FormatException catch (e) {
          throw ArgumentError('Could not parse CSV row ${data.length + 1}: ${e.message}');
        }
      }
    }

    if (data.isEmpty) {
      throw ArgumentError('CSV has no data rows to load.');
    }

    return DataSet(data: data);
  }

  /// Renders this set as a full CSV document.
  String toCsvString() {
    final codec = CsvCodec(
      eol: '\n',
      fieldDelimiter: ',',
      convertNullTo: 'null', // keep fields rendered even when empty
    );
    final rows = [
      ['Traffic_Sign', 'Path'],
      ...data.map((v) => v.toCsvRecord()),
    ];
    return codec.encoder.convert(rows, delimitAllFields: false);
  }

  /// Serialises this set and writes it to [path].
  void writeCsv(String path) => File(path).writeAsStringSync(toCsvString());
}
