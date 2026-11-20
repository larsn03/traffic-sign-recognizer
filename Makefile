# Convenience wrappers around the Traffic Sign Recognizer CLI.
#
# The recognizer is a Dart program; every target invokes it via `dart run`.
# Generated artifacts are written under ./output, mirroring the Go project
# layout (see PROJECT_SPEC.md, section 13).

APP    := dart run bin/traffic_sign_recognizer.dart
ASSETS := ./assets
OUT    := ./output

.PHONY: all prepare extract etree ctree cknn cknnlog test clean

# Run the full pipeline: extract (seeded 60/40 split), train the tree,
# classify.
all: prepare extract etree ctree cknn

# Ensure the output directory exists.
prepare:
	mkdir -p $(OUT)

# Extract all images under assets/images: a seeded random 60/40 split into
# training_data.csv and to_be_classified_data.csv (the seed is printed).
extract: prepare
	$(APP) -e $(ASSETS)/images $(OUT)/training_data.csv $(OUT)/to_be_classified_data.csv

# Train the decision tree from the training CSV (z-standardized).
etree: prepare
	$(APP) -dtree-train $(OUT)/training_data.csv $(OUT)/tree.json -z

# Classify the test set with the trained decision tree (z-standardized).
ctree: prepare
	$(APP) -dtree-classify $(OUT)/tree.json $(OUT)/to_be_classified_data.csv $(OUT)/results_tree.csv -z

# Classify the test set with KNN (default k=5, z-standardized).
cknn: prepare
	$(APP) -knn $(OUT)/to_be_classified_data.csv $(OUT)/training_data.csv -z

# KNN classification with misclassifications logged to classifying.log.
cknnlog: prepare
	$(APP) -knn $(OUT)/to_be_classified_data.csv $(OUT)/training_data.csv -z -printfaulty

# Run every non-extraction command (train + classify) without any
# standardization and with all five standardization methods. The extracted
# CSVs are only generated if missing, so the slow image extraction is
# skipped on repeat runs.
test: prepare
	@test -f $(OUT)/training_data.csv -a -f $(OUT)/to_be_classified_data.csv || $(APP) -e $(ASSETS)/images $(OUT)/training_data.csv $(OUT)/to_be_classified_data.csv
	$(APP) -dtree-train $(OUT)/training_data.csv $(OUT)/tree_raw.json
	$(APP) -dtree-classify $(OUT)/tree_raw.json $(OUT)/to_be_classified_data.csv $(OUT)/results_tree_raw.csv
	$(APP) -knn $(OUT)/to_be_classified_data.csv $(OUT)/training_data.csv
	$(APP) -dtree-train $(OUT)/training_data.csv $(OUT)/tree.json -z
	$(APP) -dtree-classify $(OUT)/tree.json $(OUT)/to_be_classified_data.csv $(OUT)/results_tree.csv -z
	$(APP) -knn $(OUT)/to_be_classified_data.csv $(OUT)/training_data.csv -z
	$(APP) -knn $(OUT)/to_be_classified_data.csv $(OUT)/training_data.csv -z -printfaulty
	$(APP) -dtree-train $(OUT)/training_data.csv $(OUT)/tree_minmax.json -minmax
	$(APP) -dtree-classify $(OUT)/tree_minmax.json $(OUT)/to_be_classified_data.csv $(OUT)/results_tree_minmax.csv -minmax
	$(APP) -knn $(OUT)/to_be_classified_data.csv $(OUT)/training_data.csv -minmax
	$(APP) -dtree-train $(OUT)/training_data.csv $(OUT)/tree_l1.json -l1
	$(APP) -dtree-classify $(OUT)/tree_l1.json $(OUT)/to_be_classified_data.csv $(OUT)/results_tree_l1.csv -l1
	$(APP) -knn $(OUT)/to_be_classified_data.csv $(OUT)/training_data.csv -l1
	$(APP) -dtree-train $(OUT)/training_data.csv $(OUT)/tree_l2.json -l2
	$(APP) -dtree-classify $(OUT)/tree_l2.json $(OUT)/to_be_classified_data.csv $(OUT)/results_tree_l2.csv -l2
	$(APP) -knn $(OUT)/to_be_classified_data.csv $(OUT)/training_data.csv -l2
	$(APP) -dtree-train $(OUT)/training_data.csv $(OUT)/tree_rs.json -rs
	$(APP) -dtree-classify $(OUT)/tree_rs.json $(OUT)/to_be_classified_data.csv $(OUT)/results_tree_rs.csv -rs
	$(APP) -knn $(OUT)/to_be_classified_data.csv $(OUT)/training_data.csv -rs

# Remove generated artifacts.
clean:
	rm -rf $(OUT) classifying.log
