# Convenience wrappers around the Traffic Sign Recognizer CLI.
#
# The recognizer is a Dart program; every target invokes it via `dart run`.
# Generated artifacts are written under ./output, mirroring the Go project
# layout (see PROJECT_SPEC.md, section 13).

APP    := dart run bin/traffic_sign_recognizer.dart
ASSETS := ./assets
OUT    := ./output

.PHONY: all prepare eclass etrain etree ctree cknn cknnlog test clean

# Run the full pipeline: extract both sets, train the tree, classify.
all: prepare eclass etrain etree ctree cknn

# Ensure the output directory exists.
prepare:
	mkdir -p $(OUT)

# Extract the test-set feature vectors.
eclass: prepare
	$(APP) -e $(ASSETS)/to_be_classified_data/ $(OUT)/to_be_classified_data.csv

# Extract the training-set feature vectors.
etrain: prepare
	$(APP) -e $(ASSETS)/training_data $(OUT)/training_data.csv

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

# Run every non-extraction command (train + classify) with all four
# standardization methods. The extracted CSVs are only generated if
# missing, so the slow image extraction is skipped on repeat runs.
test: prepare
	@test -f $(OUT)/training_data.csv || $(APP) -e $(ASSETS)/training_data $(OUT)/training_data.csv
	@test -f $(OUT)/to_be_classified_data.csv || $(APP) -e $(ASSETS)/to_be_classified_data/ $(OUT)/to_be_classified_data.csv
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

# Remove generated artifacts.
clean:
	rm -rf $(OUT) classifying.log
