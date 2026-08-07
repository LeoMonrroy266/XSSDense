#!/usr/bin/env python3
# coding:utf-8

import os
import sys
import glob
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ───────────────────────────────
# Config
# ───────────────────────────────
GRID_SIZE = 52  # change if needed

# ───────────────────────────────
# TFRecord helpers
# ───────────────────────────────
def serialize_example(flat_array, index):
    return tf.train.Example(features=tf.train.Features(feature={
        "data": tf.train.Feature(float_list=tf.train.FloatList(value=flat_array)),
        "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
    })).SerializeToString()

def write_tfrecord(filename, examples):
    with tf.io.TFRecordWriter(filename) as writer:
        for ex in tqdm(examples, desc=f"Writing {filename}"):
            writer.write(ex)

def normalize_signed(data, M):
    return (data / M).astype(np.float32)

# ───────────────────────────────
# Main workflow
# ───────────────────────────────
def run_workflow(npy_dir, reference_path, output_dir, num_threads=8):

    tf_out = os.path.join(output_dir, "tfrecords")
    os.makedirs(tf_out, exist_ok=True)

    # Load reference
    print("Loading reference...")
    ref = np.load(reference_path)

    # Load all npy files
    npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    npy_files = [f for f in npy_files if os.path.abspath(f) != os.path.abspath(reference_path)]

    if len(npy_files) == 0:
        raise RuntimeError("No npy files found.")

    print(f"Found {len(npy_files)} samples")

    # Compute differences
    print("Computing differences...")
    raw_diffs = []
    names = []

    for f in tqdm(npy_files):
        arr = np.load(f)

        if arr.shape != ref.shape:
            raise ValueError(f"Shape mismatch: {f} {arr.shape} != {ref.shape}")

        d = arr - ref
        raw_diffs.append(d)
        names.append(os.path.basename(f))

    # Compute normalization constant
    print("Computing normalization...")
    M = max(max(abs(d.min()), abs(d.max())) for d in raw_diffs)

    if M <= 0:
        raise RuntimeError("Normalization constant M is non-positive.")

    diff_norm_list = [normalize_signed(d, M) for d in raw_diffs]

    # Serialize
    print("Serializing TFRecords...")
    examples_train, examples_test = [], []

    def process_sample(idx):
        grid = diff_norm_list[idx]

        # ensure correct shape (no pad needed if already correct)
        if grid.shape != (GRID_SIZE, GRID_SIZE, GRID_SIZE):
            raise ValueError(f"Unexpected shape: {grid.shape}")

        flat = grid.flatten()
        return idx, serialize_example(flat, idx)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(
            executor.map(process_sample, range(len(diff_norm_list))),
            total=len(diff_norm_list)
        ))

    for idx, ex in results:
        if idx % 5 == 0:
            examples_test.append(ex)
        else:
            examples_train.append(ex)

    write_tfrecord(os.path.join(tf_out, "train.tfrecords"), examples_train)
    write_tfrecord(os.path.join(tf_out, "test.tfrecords"), examples_test)

    # Save normalization params
    np.savez(os.path.join(tf_out, "signed_norm_params.npz"),
             type="signed", M=np.array([M], dtype=np.float64))

    # Metadata
    meta = {
        "grid_size": GRID_SIZE,
        "reference": os.path.basename(reference_path),
        "n_samples": len(diff_norm_list),
        "M": float(M)
    }

    with open(os.path.join(tf_out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print(f"TFRecords saved to: {tf_out}")
    print(f"Normalization M: {M:.6g}")

# ───────────────────────────────
# CLI
# ───────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <npy_dir> <reference.npy> <output_dir>")
        sys.exit(1)

    npy_dir = sys.argv[1]
    reference_path = sys.argv[2]
    output_dir = sys.argv[3]

    run_workflow(npy_dir, reference_path, output_dir)
