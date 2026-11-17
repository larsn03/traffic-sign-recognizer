# traffic_sign_recognizer

A sample Dart command-line application that ported the Go project
[`traffic_sign_recognizer`](../traffic_sign_recognizer) into a pure-Dart code
base. It implements two `traffic-sign` classifiers and lets them train on
labelled feature vectors, classify held-out data, persist models to disk, and
report per-entry verdicts plus an overall accuracy score.

The project contains:

- library code in [`lib/`](lib/) — the reusable `traffic_sign_recognizer`
  package (data model, CSV codec, CART decision tree, KNN classifier),
- a runnable demo pipeline in
  [`bin/traffic_sign_recognizer.dart`](bin/traffic_sign_recognizer.dart), and
- an example unit test in
  [`test/data_set_test.dart`](test/data_set_test.dart).

## Requirements

- A Dart SDK of at least version `3.13.7`. The bundled Flutter `dart` tooling is
  used by the commands below; any recent Dart command-line tool works too.

## Build / compile

The app has no build system — it runs straight from source on top of the Dart
SDK, so "compiling" just means pulling dependencies and running the entrypoint.

Install dependencies (resolve `pubspec.yaml`):

```bash
dart pub get
```

Compile a release snapshot:

```bash
dart compile exe bin/traffic_sign_recognizer.dart -o tsr
./tsr
```

You can also run the demo directly from source without compiling first:

```bash
dart run bin/traffic_sign_recognizer.dart
```

The commands above assume a normal Dart installation. In this repo's environment
Flutter ships a wrapper `dart` that sometimes fails with an `engine.stamp`
permission error; if so, use the underlying SDK binary directly, e.g.:

```bash
/usr/share/flutter/bin/cache/dart-sdk/bin/dart run bin/traffic_sign_recognizer.dart
```

## Run the demo

With no command-line arguments required:

```bash
dart run bin/traffic_sign_recognizer.dart
```

The demo does **not** depend on any CSV data files. It builds a tiny synthetic
training set of two tightly grouped classes (`Sign_A` vs `Sign_B`) in
`buildTrainingData()`, runs each classifier against the held-out test vectors in
`buildTestData()`, and prints a report for every model:

- **KNN classification (k=3)** — feature standardization before distance queries.
- **Decision-tree classification** — trained with `learnFromDataSet`.
- **Decision-tree classification (from tree.json)** — retrains from the saved
  JSON file to prove persistence works.

## Run the tests

```bash
dart test
```

The example (`test/data_set_test.dart`) verifies that a feature CSV round-trips
through the codec and the data-set loader, confirming the `Traffic_Sign`/`Path`
schema survives an in-memory → string → in-memory cycle.

## Static analysis

Lint settings live in [`analysis_options.yaml`](analysis_options.yaml)
(inheriting `package:lints/recommended`). Run:

```bash
dart analyze
```

To also run the newer, more precise Dart linter tool:

```bash
dart pub global activate dartlint
dartlint --fix --report all
```

## Programmatic usage

Import the library and use its classifiers:

```dart
import 'package:traffic_sign_recognizer/traffic_sign_recognizer.dart';

void main() {
  // Train on labelled data.
  final train = DataSet(
    data: [
      DataVector.fromNumericFeatures('Sign_A', 'a.jpg', <double>[...]),
      DataVector.fromNumericFeatures('Sign_B', 'b.jpg', <double>[...]),
    ],
  );

  // KNN classifier with feature standardization.
  final knn = KnnClassifier(alreadyClassifiedData: train, k: 3);
  final knnResults = knn.classify(testSet);
  print(knnResults.total);      // total entries classified
  print(knnResults.accuracy);   // percentage correct (0–100)

  // Decision-tree classifier: learn, persist, reload, classify.
  final dt = DecisionTreeClassifier();
  final tree = dt.learnFromDataSet(train);
  File('tree.json').writeAsStringSync(tree.toJsonString());
  final reloaded = DataTree.loadFromFile('tree.json');
  final treeResults = reloaded.classify(testSet);

  // Report + outputs.
  print(treeResults.toStringLines().join('\n'));
  File('results.csv').writeAsStringSync(treeResults.toCsvString());
}
```

### Public API reference

| Type | Purpose |
| --- | --- |
| `DataVector` | A single labelled sample (43-feature layout) with CSV read/write and feature helpers. |
| `DataSet` | A collection of vectors; loads from CSV and exposes a filter for classification use. |
| `DataTree` | CART decision tree: `learn`, `classify`, JSON serialization, persisted to disk. |
| `KnnClassifier` | K-Nearest-Neighbours classifier with standardization + majority voting. |
| `DecisionTreeClassifier` | Adapter exposing both the classifier and learner APIs on top of a `DataTree`. |
| `ClassificationResult` | One verdict: expected vs. predicted label, plus `stringify()`. |
| `ClassificationResultSet` | Aggregate verdicts with `total`, `accuracy`, `faulty`, CSV/log rendering. |

The two classifier adapters satisfy the documented Dart interfaces in
[`lib/classifier/`](lib/classifier), so they expose both a "classify" entry point
and an optional "learn" (`learner`) entry point.

## Output files

Running the demo writes three files into your working directory:

- `results.csv` — one row per classified vector (`path,shouldBe,predicted`).
- `faulty.log` — the vectors each model got wrong (append mode).
- `tree.json` — the trained decision tree in the internal JSON format.

## License

Provided as an educational sample; see the upstream project's license if you
reuse its sources.
