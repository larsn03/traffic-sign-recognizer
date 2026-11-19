# Traffic Sign Recognizer (Dart)

A command-line tool that extracts hand-crafted numerical features from images
of German traffic signs, stores them as CSV feature vectors, and classifies
them into one of six sign categories using two interchangeable algorithms:

- **K-Nearest-Neighbors (KNN)** over z-score-standardized Euclidean distance.
- A **decision tree** trained with batch CART / Gini impurity.

This is a faithful Dart port of the original Go project. See
[`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the complete, language-agnostic
specification that this implementation follows.

## The six classes

| Label | Meaning |
|---|---|
| `Stop` | Stop |
| `Vorfahrt gewähren` | Give way / yield |
| `Vorfahrtsstraße` | Priority road |
| `Vorfahrt von rechts` | Yield to the right |
| `Fahrtrichtung links` | Turn left |
| `Fahrtrichtung rechts` | Turn right |

## Requirements

- **Dart SDK** `^3.13.7` (developed against 3.14).
- Dependencies (resolved by `dart pub get`): `csv`, `image`, `path`.

## Setup

```bash
dart pub get
```

## Usage

The entrypoint is `bin/traffic_sign_recognizer.dart`. All modes are invoked as:

```bash
dart run bin/traffic_sign_recognizer.dart <mode> [args]
```

### Extract features → CSV

```bash
dart run bin/traffic_sign_recognizer.dart -e <imageDir> <out.csv>
```

Walks `<imageDir>` recursively; each subfolder name is the class label. Writes
one feature-vector row per image (45 columns; see [CSV schema](#csv-schema)).

### KNN classification

```bash
dart run bin/traffic_sign_recognizer.dart -knn <test.csv> <train.csv> [k] [-z]
```

Classifies the test set using the training set. `k` defaults to `5`. Prints one
result line per image plus `Correctly classified: N%`. Test sets with more
than 1000 rows are classified by a pool of worker isolates (see
[Parallelism](#parallelism)); results are identical to a sequential run.

Feature scaling is controlled by the `-z` flag (see
[Standardization](#standardization)).

### Decision-tree training

```bash
dart run bin/traffic_sign_recognizer.dart -dtree-train <train.csv> <tree.json>
```

Trains a batch-CART tree and serializes it to `<tree.json>`. Training features
are scaled by the [data mask](#standardization).

### Decision-tree classification

```bash
dart run bin/traffic_sign_recognizer.dart -dtree-classify <tree.json> <test.csv> <out.csv>
```

Loads the tree, classifies the test set, and writes one prediction per line to
`<out.csv>`. Test features are scaled by the same [data mask](#standardization)
that was used during training.

### `-printfaulty`

Append `-printfaulty` to a KNN or decision-tree classification to also write
the misclassified images to `classifying.log`.

### Result line format

```
Image <path> | Should be <label> | Classified as <label>
```

## Standardization

How features are scaled before classification:

- **KNN with `-z`** — z-score standardization: each of the 43 features is
  transformed to `x → (x − mean) / stddev`, where `mean`/`stddev` are fitted on
  the training set only (population standard deviation; zero-variance features
  map to `0`).
- **KNN without `-z` (default)** — no z-scoring; each feature is multiplied by
  its [data mask](#data-mask) factor instead.
- **Decision tree** — always uses the data mask (it has no z-score path); the
  mask is applied both when training and when predicting.

Extraction never applies any scaling; CSV files always contain raw features.

### Data mask

The data mask is a YAML file with one factor per feature *family* (a group of
related parameters):

| family                       | features              |
|------------------------------|-----------------------|
| `number_of_different_colors` | color count           |
| `contains_colors`            | red/blue/yellow/black flags |
| `number_of_changes_vertically`   | 10 vertical change counts |
| `number_of_changes_horizontally` | 10 horizontal change counts |
| `white_fraction`             | 9 white fractions     |
| `left_right_asymmetry`       | 9 asymmetries         |

Every feature of a data vector is multiplied by its family's factor before
classification. The mask is read from `output/data-mask.yaml`; if the file does
not exist it is created there with every factor `1.0` (the identity, i.e. no
scaling). See `data-mask-example.yaml` for the format. To change the scaling,
edit the factors in `output/data-mask.yaml` and re-run (retrain the tree if you
change the mask between training and classification).

## Makefile

Convenience wrappers around the CLI (artifacts are written under `./output`):

| Target | Command |
|---|---|
| `eclass` | extract the test set → `output/to_be_classified_data.csv` |
| `etrain` | extract the training set → `output/training_data.csv` |
| `etree` | train the tree → `output/tree.json` |
| `ctree` | classify the test set with the tree → `output/results_tree.csv` |
| `cknn` | classify the test set with KNN (data-mask scaling) |
| `cknnz` | classify the test set with KNN + `-z` (z-score standardization) |
| `cknnlog` | KNN + `-printfaulty` (logs to `classifying.log`) |
| `all` | run the full pipeline |
| `clean` | remove generated artifacts |

```bash
make all
```

## Dataset splitter

`mix_datasets.py` splits a labelled dataset 60/40 into
`training_data_new` / `to_be_classified_data_new` (seed `42`, structure
preserved, source untouched):

```bash
python3 mix_datasets.py <source_dir> <output_dir>
```

## Parallelism

Extraction uses an isolate work-stealing pool. On machines where a single
process cannot saturate all cores, large jobs are automatically sharded across
child processes.

KNN classification parallelises the same way: every test row is classified
independently, so rows are handed one at a time to a pool of worker isolates
(test sets with more than 1000 rows). Results are stored by row index, so the
output order and every prediction are identical to a sequential run — only
the wall time changes (on this machine: ~2 min → ~14 s for the 14.5k-row test
set).

Both can be controlled with environment variables:

- `TSR_WORKERS` — total worker isolates (default: CPU count).
- `TSR_SHARDS` / `TSR_SHARD` — internal sharding coordinates for extraction
  (set automatically by the orchestrator; generally not needed by hand).

## CSV schema

45 columns per row: `Traffic_Sign`, `Path`, an integer color count, four `0/1`
presence flags, ten vertical + ten horizontal color-change counts (stored as
`int/10`, one decimal), nine white fractions (`[0,1]`, two decimals), and nine
left–right asymmetries (`[-1,1]`, two decimals).

## Project layout

```
bin/traffic_sign_recognizer.dart   # CLI entrypoint
lib/
  data/                            # DataVector, DataVectorSet (CSV), DataTree
  classifier/                      # KNN + decision-tree classifiers
  extraction/                      # color classification + feature extraction
test/                              # unit + integration tests
assets/
  training_data/                   # labelled training images (class subfolders)
  to_be_classified_data/           # images to classify
  all_signs_png/                   # one reference PNG per sign
```

## Tests

```bash
dart test
```
