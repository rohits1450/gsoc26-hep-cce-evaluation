#!/usr/bin/env python3
"""
Phase 1: Synthetic dataset generation.
Creates .npz shards of fake image data (224x224x3) to use for the DataLoader benchmark.

Usage:
    python generate_dataset.py [--output_dir ./synthetic_data] [--total_gb 5] [--seed 42]
"""

import os
import argparse
import time
import numpy as np


def generate_dataset(output_dir: str, total_gb: float, seed: int):
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(seed)

    sample_shape = (224, 224, 3)
    bytes_per_sample = int(np.prod(sample_shape))  # 150,528 bytes
    samples_per_file = 3500  # around 500 MB per file
    bytes_per_file = bytes_per_sample * samples_per_file

    total_bytes = int(total_gb * 1024**3)
    num_files = max(1, total_bytes // bytes_per_file)

    print(f"output_dir={output_dir}, target={total_gb}GB, files={num_files}, "
          f"samples/file={samples_per_file}, seed={seed}")
    print()

    total_samples = 0
    total_written = 0
    start_time = time.time()

    for i in range(num_files):
        file_path = os.path.join(output_dir, f"shard_{i:04d}.npz")

        images = rng.randint(0, 256, size=(samples_per_file, *sample_shape), dtype=np.uint8)
        labels = rng.randint(0, 1000, size=(samples_per_file,), dtype=np.int64)

        np.savez(file_path, images=images, labels=labels)

        file_size = os.path.getsize(file_path)
        total_written += file_size
        total_samples += samples_per_file

        elapsed = time.time() - start_time
        print(f"  [{i+1}/{num_files}] {file_path} "
              f"({file_size / 1024**2:.1f} MB) "
              f"[{elapsed:.1f}s, {total_written / 1024**3:.2f} GB written]")

    elapsed = time.time() - start_time
    print(f"\ndone — {num_files} files, {total_samples:,} samples, "
          f"{total_written / 1024**3:.2f} GB in {elapsed:.1f}s "
          f"({total_written / 1024**2 / elapsed:.1f} MB/s)")

    # save metadata so benchmark.py knows shard layout
    metadata = {
        "num_files": num_files,
        "samples_per_file": samples_per_file,
        "total_samples": total_samples,
        "sample_shape": list(sample_shape),
        "num_classes": 1000,
        "seed": seed,
    }
    meta_path = os.path.join(output_dir, "metadata.npz")
    np.savez(meta_path, **{k: np.array(v) for k, v in metadata.items()})
    print(f"metadata -> {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./synthetic_data")
    parser.add_argument("--total_gb", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.output_dir, args.total_gb, args.seed)
