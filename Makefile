# Targets mirror the `traffic_sign_recognizer` Go CLI (see ../Makefile).
# Each one forwards flags + paths to bin/traffic_sign_recognizer.dart, which
# parses them like the Go `run()` and dispatches to a single mode. The targets
# are ordered so building any of them pulls in the extraction step it needs:
#   ctree  -> etree -> etrain;  eclass  (standalone).

# Resolve `dart`, preferring the bundled Flutter SDK binary — the `flutter`
# wrapper hits an engine.stamp permission error here (see README). Override on
# the command line with `DART=... make ...`.
DART ?= $(shell command -v dart 2>/dev/null || true)
ifeq ($(strip $(DART)),)
	DART := /usr/share/flutter/bin/cache/dart-sdk/bin/dart
endif

# Extract feature data from the classifiable assets into a CSV.
eclass:
	@echo "== Extract classifiable data from assets =="
	@mkdir -p output
	$(DART) run bin/traffic_sign_recognizer.dart \
		-e ./assets/to_be_classified_data/ ./output/to_be_classified_data.csv

# Extract the training images into a CSV; depends on nothing.
etrain:
	@echo "== Extract training data from assets =="
	@mkdir -p output
	$(DART) run bin/traffic_sign_recognizer.dart \
		-e ./assets/training_data ./output/training_data.csv

# Train a decision tree on the extracted training data and save it as JSON.
etree: etrain
	@echo "== Train decision tree =="
	$(DART) run bin/traffic_sign_recognizer.dart \
		-dtree-train ./output/training_data.csv ./tree.json

# Classify the held-out set with the trained tree, writing verdicts to disk.
ctree: etree eclass
	@echo "== Classify with decision tree =="
	$(DART) run bin/traffic_sign_recognizer.dart \
		-dtree-classify ./tree.json ./output/to_be_classified_data.csv ./results_tree.csv

# KNN classification of the held-out set using the labelled training set.
cknn:
	@echo "== KNN classification (k=3) =="
	@mkdir -p output
	$(DART) run bin/traffic_sign_recognizer.dart \
		-knn ./output/to_be_classified_data.csv ./output/training_data.csv

# KNN classification that also dumps misclassified rows to classifying.log.
cknnlog:
	@echo "== KNN classification (faulty log) =="
	$(DART) run bin/traffic_sign_recognizer.dart \
		-knn -printfaulty ./output/to_be_classified_data.csv ./output/training_data.csv

.PHONY: eclass etrain etree ctree cknn cknnlog
