#!/usr/bin/env python3
"""Split a labelled traffic-sign dataset into 60/40 training/test sets.

Walks each class subfolder of the source dataset, shuffles the images with a
fixed seed, and copies 60% into ``<out>/training_data_new`` and 40% into
``<out>/to_be_classified_data_new``, preserving the relative folder structure.
The source dataset is left untouched and a per-class summary is printed.

This is a pure-stdlib script (``os``, ``random``, ``shutil``) mirroring the
Go project's ``mix_datasets.py`` (PROJECT_SPEC.md, section 11.3).

Usage:
    python3 mix_datasets.py <source_dir> <output_dir>
"""

import os
import random
import shutil
import sys

SEED = 42
TRAIN_RATIO = 0.6
TRAIN_NAME = "training_data_new"
TEST_NAME = "to_be_classified_data_new"


def collect(source_dir):
    """Return a list of ``(abs_path, rel_path)`` for every file under each class.

    ``rel_path`` is relative to ``source_dir`` and therefore starts with the
    class folder name, which preserves the per-class structure on copy.
    """
    items = []
    for class_name in sorted(os.listdir(source_dir)):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for root, _dirs, files in os.walk(class_dir):
            for name in sorted(files):
                abs_path = os.path.join(root, name)
                rel_path = os.path.relpath(abs_path, source_dir)
                items.append((abs_path, rel_path))
    return items


def copy_subset(items, output_dir, dest_name):
    """Copy ``items`` into ``<output_dir>/<dest_name>`` keeping rel structure.

    Returns a per-class count of the copied files.
    """
    dest_root = os.path.join(output_dir, dest_name)
    per_class = {}
    for abs_path, rel_path in items:
        dest_path = os.path.join(dest_root, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(abs_path, dest_path)
        class_name = rel_path.split(os.sep)[0]
        per_class[class_name] = per_class.get(class_name, 0) + 1
    return per_class


def main(argv):
    if len(argv) != 3:
        print(f"usage: {argv[0]} <source_dir> <output_dir>", file=sys.stderr)
        return 1

    source_dir = argv[1]
    output_dir = argv[2]
    if not os.path.isdir(source_dir):
        print(f"error: source dir not found: {source_dir}", file=sys.stderr)
        return 1

    items = collect(source_dir)
    if not items:
        print(f"error: no files found under {source_dir}", file=sys.stderr)
        return 1

    random.seed(SEED)
    random.shuffle(items)

    split = int(len(items) * TRAIN_RATIO)
    train_items = items[:split]
    test_items = items[split:]

    os.makedirs(output_dir, exist_ok=True)
    train_counts = copy_subset(train_items, output_dir, TRAIN_NAME)
    test_counts = copy_subset(test_items, output_dir, TEST_NAME)

    print(
        f"Split {len(items)} images "
        f"(seed={SEED}, ratio={TRAIN_RATIO:.0%} train / {1 - TRAIN_RATIO:.0%} test)"
    )
    for class_name in sorted(set(train_counts) | set(test_counts)):
        t = train_counts.get(class_name, 0)
        e = test_counts.get(class_name, 0)
        print(f"  {class_name}: {t} train, {e} test (total {t + e})")
    print(
        f"Totals: {len(train_items)} -> {TRAIN_NAME}, "
        f"{len(test_items)} -> {TEST_NAME}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
