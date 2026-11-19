# Convenience wrappers around the Traffic Sign Recognizer CLI.
#
# The recognizer is a Dart program; every target invokes it via `dart run`.
# Generated artifacts are written under ./output, mirroring the Go project
# layout (see PROJECT_SPEC.md, section 13).

APP    := dart run bin/traffic_sign_recognizer.dart
ASSETS := ./assets
OUT    := ./output

.PHONY: all prepare eclass etrain etree ctree cknn cknnz cknnlog clean

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

# Train the decision tree from the training CSV.
etree: prepare
	$(APP) -dtree-train $(OUT)/training_data.csv $(OUT)/tree.json

# Classify the test set with the trained decision tree.
ctree: prepare
	$(APP) -dtree-classify $(OUT)/tree.json $(OUT)/to_be_classified_data.csv $(OUT)/results_tree.csv

# Classify the test set with KNN (default k=5, data-mask scaling).
cknn: prepare
	$(APP) -knn $(OUT)/to_be_classified_data.csv $(OUT)/training_data.csv

# Classify the test set with KNN using z-score standardization (-z).
cknnz: prepare
	$(APP) -knn $(OUT)/to_be_classified_data.csv $(OUT)/training_data.csv -z

# KNN classification with misclassifications logged to classifying.log.
cknnlog: prepare
	$(APP) -knn $(OUT)/to_be_classified_data.csv $(OUT)/training_data.csv -printfaulty

# Remove generated artifacts.
clean:
	rm -rf $(OUT) classifying.log
