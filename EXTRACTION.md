# Data Extraction — from Images to Data Vectors to CSVs

A complete, code-free description of how traffic-sign images are turned into feature vectors (data vectors) and then written to CSV files. The goal is that a coding agent can rebuild the entire pipeline from this document alone. Everything here describes **logic**, not implementation.

---

## 0. End-to-end flow

```
Images on disk (class folders)
        │   read classes + collect images
        ▼
One feature vector per image   ← decode, color analysis, frame removal, geometry
        │   DataVectorV1 build
        ▼
CSV file (feature matrix)  →  used as training/classification input
```

The pipeline has three conceptual stages:

1. **Discovery** — read the input directory, gather all images grouped by class.
2. **Per-image extraction** — turn each single image into a data vector holding a fixed set of features.
3. **Serialization** — write every successfully extracted vector as one row in a CSV.

The output CSV is what later stages (K-NN classification, decision-tree classification) consume. `traffic_sign_recognizer -e ./images/dir ./out.csv` runs the whole flow; relative paths for both input and output are supported.

---

## 1. Input: the images and their directory structure

The input is a folder that contains one subfolder per traffic-sign class. The subfolder **name is the label** (e.g. `Stop`, `Vorfahrt gewähren`). Within each class, image files may live at any depth — the extractor searches recursively. It accepts `.png`, `.jpg`/`.jpeg`, `.bmp`, `.gif` and `.tif`/`.tiff` files.

A representative real layout looks like:

```
<pathToBeExtracted>/
├── Fahrtrichtung links/          # class label
│   └── -1/80x60+3/X-10Y-30.bmp    # image at arbitrary depth
├── Stop/
│   └── +2/80x60+2/X-30Y-80.bmp
├── Vorfahrt gewähren/
│   └── +2/80x60+2/X80Y50.bmp
├── Vorfahrtsstraße/
│   └── +1/80x60+1/X-80Y-30.bmp
└── ... (other classes)
```

Notes:

- **Classes are the top-level directories.** Only directories become class labels; files at the top level of the folder are ignored.
- **Depth is irrelevant.** The same class can have images spread across many nested subfolders (e.g. `-1/80x60+1`, `+2/80x60`, a leaf directory with a numeric name such as `3500`). All of them are collected.
- **Exactly one sign per image.** Each image is drawn on a plain white background with a single traffic sign; this uniformity is what makes the "white frame removal" step meaningful.
- **Signs can have up to 5 possible colors** and there are (around) six signs in the dataset — the extractor itself does not need to know these counts; it derives classes purely from folder names.

The directory structure of the input is expected to stay as-is for rebuild purposes. The relative path of each image (relative to the input folder) is recorded so any extracted vector can be traced back to its source file.

---

## 2. File discovery

1. **Read classes.** List the top-level entries of the input folder; every entry that is a directory becomes a class label. The class list is sorted alphabetically so output ordering is deterministic.
2. **Collect images per class.** For each class folder, recursively walk its tree and keep files whose extension is one of: `.bmp`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.tif`, `.tiff`. The per-class file list is sorted by path (recursive / full-path sort).
3. **Build jobs.** Each image becomes a *job* triple: `{ class label, absolute path, relative path }`. A job is what gets handed to the extractor.

Order of work follows classes in alphabetical order, then images within each class in sorted-path order. Output rows are emitted in this same original order even though extraction runs concurrently (see §4).

---

## 3. Per-image feature extraction

Every image becomes a single **data vector** via a fixed sequence of steps:

1. **Decode.** Read the file with any standard image decoder and normalize the result into a uniform RGBA representation so all downstream logic can treat every image identically.
2. **Color analysis (full image).** Over the entire, uncropped image, classify every pixel and record which color categories are present. This uses the full image bounds (no crop).
3. **White-frame removal.** Find the bounding box of non-white pixels — everything outside is treated as white background/frame and excluded from the geometry steps.
4. **Crop.** Restrict all subsequent geometry to the inner rectangle found in step 3.
5. **Geometry features.** On the inner (cropped) region, compute four feature groups: vertical line counts, horizontal line counts, white-fraction per grid cell, and left–right asymmetry per grid cell.

If decoding or opening fails for an image, a warning is emitted to standard error and the image is skipped — no row is written for it.

Work across images is parallelized with a worker pool (one worker per available CPU thread), jobs distributed via channels. Results are returned in the original job order regardless of completion order.

---

## 4. Color classification

Every pixel is assigned to exactly one of five color categories plus an `unknown` fallback: **red, yellow, blue, white, black**. A pixel that fits none of them stays `unknown`.

1. **Convert** the pixel's RGBA to HSV — hue in degrees (0–360), saturation and value on a 0–255 scale.
2. **Achromatic resolution first:** if value is low (< 45) → *black*; otherwise if saturation is low (< 45) and value is high (> 150) → *white*.
3. **Hue/saturation/value bands** (used when not already resolved as black/white):

| Category | HSV condition | Notes |
|----------|---------------|-------|
| Red      | hue < 20 or hue > 340, S > 50, V > 50 | wraps around 0°; captures red-to-orange |
| Blue     | hue ∈ [190, 250], S > 45, V > 45 | deep to light blue |
| Yellow   | hue ∈ [28, 60], S > 25, V > 55 | yellow; warm sign-yellow tolerated |
| Black    | V < 45 (any H) | dark / achromatic |
| White    | S < 45 and V > 150 | bright achromatic background/fill |

Red is treated as wrapping around 0°, so it also captures slightly orange-red hues. `unknown` pixels are silently ignored in all feature logic.

### Aggregated color features (per image)
- **Number of different colors** — how many of {red, yellow, blue, white, black} appear at least once anywhere in the full image; range 0–5.
- **Contains_red / contains_blue / contains_yellow / contains_black** — boolean presence flags (1/0), each true if that color appears anywhere in the full image.

---

## 5. White-frame removal (cropping)

The background is plain white, so the sign's own bounding box can be found by looking for non-white content:

- Scan every pixel of the full image; track the minimum and maximum coordinates among the **non-white** pixels (`isWhitePixel` = all three RGB channels bright, > 180, and roughly balanced).
- Emit the inner rectangle as `(minX, minY, maxX+1, maxY+1)` — an exclusive upper bound. The `+1` margins keep a one-pixel edge of content so a thin sign border is not clipped away.
- **Fallback:** if no non-white pixels exist (a fully white image), fall back to the full bounds so the downstream detectors yield deterministic zero counts.

The inner rectangle defined here feeds every subsequent geometric feature. Note that detection of "white" for frame removal uses the channel-brightness rule, which is independent of the HSV-based `white` color category used elsewhere.

---

## 6. Geometry features (on the cropped inner region)

Four groups are computed on the inner rectangle: vertical line counts, horizontal line counts, white fractions per cell, and left–right asymmetry per cell. Each group becomes one set of CSV columns.

### 6a. Vertical color-change lines (`Number_of_color_changes_vertically_*`)
- Divide the inner width into **11 equal intervals** (relative positioning, not absolute pixels).
- Place **ten vertical lines** at the interval boundaries; each is scanned top-to-bottom.

### 6b. Horizontal color-change lines (`Number_of_color_changes_horizontally_*`)
- Divide the inner height into **11 equal intervals**.
- Place **ten horizontal lines** at the interval boundaries; each is scanned left-to-right.

**Change counting (applies to both axes):**
1. Scan a line as consecutive **segments** of fixed length (segment size ≈ `lineLength / 100`, minimum 1 pixel).
2. Average all pixel colors along each segment to get that segment's representative color.
3. Classify the averaged segment color using the same HSV rule as individual pixels (§4) — no thresholding, the raw averaged value decides.
4. Compare consecutive segments; if a segment's category differs from the previous one, increment a change counter. Segment end positions are clamped to the line length. A white-to-white transition between adjacent segments counts as *no* change.

If the dimension being scanned is too short (a length of < 2) the corresponding counts stay all zeros.

### 6c. White fraction per grid cell (`White_fraction_*`)
- Split the inner region into an evenly spaced **3×3 grid** of nine cells.
- For each cell, compute `whitePixels / totalPixels` using the channel-brightness white rule — a fraction in `[0, 1]`.

### 6d. Left–right asymmetry per grid cell (`Left_right_asymmetry_*`)
- Uses the same **3×3 grid**.
- For each cell, split it vertically into left and right halves; compute the white fraction of each half (same white rule).
- Feature = `leftWhiteFraction − rightWhiteFraction`, a signed value roughly in `[-1, 1]`. Positive means white mass leans right; negative means it leans left.

Both grid sets are reported cell-by-cell, read left-to-right top-to-bottom, so the nine columns of one group align with the nine columns of the other by cell index.

---

## 7. The data vector (DataVectorV1)

Each image maps to one **data vector** — a single record bundling every extracted feature. It has exactly these fields:

| Field | Kind | Meaning |
|-------|------|---------|
| `trafficSign` | label string | class name of the source folder |
| `picturePath` | relative path | how to locate the original image again |
| `numberOfDifferentColors` | int (0–5) | distinct color categories present in the full image |
| `containsRed` / `containsBlue` / `containsYellow` / `containsBlack` | bool | presence flags of each primary color across the full image |
| `numberOfColorChangesVertically` | [10]int | vertical line change counts (§6a) |
| `numberOfColorChangesHorizontally` | [10]int | horizontal line change counts (§6b) |
| `whiteFractions` | [9]float64 | white fraction per 3×3 cell, ∈[0,1] (§6c) |
| `leftRightAsymmetry` | [9]float64 | left-minus-right white asymmetry per cell, ≈[-1,1] (§6d) |

The vector is the single unit that is serialized. Its constructor takes the label, path, color count, four presence flags, the two line-count arrays, and the two grid arrays — nothing more, nothing less. Exactly one element of each fixed-size array exists (ten lines per axis, nine cells per grid), so every row has a constant width.

---

## 8. CSV output

Every successfully extracted vector becomes exactly one CSV row; failed images produce no row at all. A fixed header names the columns in order:

1. `Traffic_Sign` — the class label (`trafficSign`).
2. `Path` — relative image path (`picturePath`).
3. `Number_of_different_colors` — integer 0–5.
4. `Contains_red` / `Contains_blue` / `Contains_yellow` / `Contains_black` — booleans as `1`/`0`.
5. `Number_of_color_changes_vertically_0..9` — the ten vertical counts. (Integer counts are formatted to one decimal place.)
6. `Number_of_color_changes_horizontally_0..9` — the ten horizontal counts (one decimal).
7. `White_fraction_0..8` — nine grid fractions in `[0, 1]`, two decimals.
8. `Left_right_asymmetry_0..8` — nine grid asymmetries in `[-1, 1]`, two decimals.

That yields **39 feature columns** per row (2 + 4 + 10 + 10 + 9 + 9), and the header matches them one-for-one. Rows are written in original job order (classes alphabetical, images by sorted path within each class). Output/input paths may be relative.

### Output artifacts produced downstream of extraction
Once the feature matrix exists as CSV, other stages consume it:

- **K-NN classification** reads one CSV (`be_classified_data.csv`) and writes a second (`already_classified_data.csv` with an added label column). It prints `Correctly classified: 90%` style accuracy; `k` defaults unless passed on the command line. With `-printfaulty`, misclassified rows are appended to `classifying.log` in the form
  `Image <path> | Should be <correct_sign> | Classified as <predicted_sign>`.
- **Decision-tree classification** produces an ordered `results_tree.csv` (one row per sample) recording `Image <path> | Should be <correct_sign> | Classified as <predicted_sign>` for each prediction.

The extractor itself only needs to emit the feature matrix CSV; the two classifiers operate on that output.

---

## 9. Rebuild checklist

- Input dirs = classes; search images recursively; sort both classes and per-class files for deterministic order.
- One data vector per image, built decode → full-image color analysis → white-frame crop → vertical/horizontal line counts → 3×3 white-fraction cells → 3×3 left/right asymmetry cells.
- Two distinct definitions of "white" exist and must not be confused: the HSV `white` category (color flags) vs. the channel-brightness rule (`>180`) used for frame removal and grid computations.
- Line detection averages RGB per segment before classifying; red wraps around 0°; white-to-white transitions are not changes.
- Every vector is a fixed-width row with exactly 39 feature columns, matching the header order above.
- Parallel extraction uses one goroutine per CPU thread but preserves original ordering in output.
- Dependencies: Go standard library only, plus `golang.org/x/image/bmp`. No external ML libraries are needed for extraction itself.
