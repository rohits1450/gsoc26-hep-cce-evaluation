#!/usr/bin/env python3
"""
Benchmark PyTorch DataLoader with varying num_workers on synthetic .npz dataset.
Tracks epoch time, throughput, and scaling efficiency.

Usage:
    python benchmark.py [--data_dir ./synthetic_data] [--batch_size 64] [--workers 1 2 4 8]
"""

import os
import time
import argparse
import json
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticFileDataset(Dataset):
    """Lazy-loading dataset from .npz shards."""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.shard_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("shard_") and f.endswith(".npz")
        ])

        if not self.shard_files:
            raise FileNotFoundError(f"No shard files found in {data_dir}")

        meta_path = os.path.join(data_dir, "metadata.npz")
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            self.samples_per_file = int(meta["samples_per_file"])
        else:
            with np.load(self.shard_files[0]) as data:
                self.samples_per_file = data["images"].shape[0]

        self.num_shards = len(self.shard_files)
        self.total_samples = self.num_shards * self.samples_per_file

        # simple per-worker cache to avoid reloading same shard
        self._cached_shard_idx = -1
        self._cached_images = None
        self._cached_labels = None

    def _load_shard(self, shard_idx: int):
        if shard_idx != self._cached_shard_idx:
            data = np.load(self.shard_files[shard_idx])
            self._cached_images = data["images"]
            self._cached_labels = data["labels"]
            self._cached_shard_idx = shard_idx

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        shard_idx = idx // self.samples_per_file
        local_idx = idx % self.samples_per_file

        self._load_shard(shard_idx)

        image = self._cached_images[local_idx]
        label = int(self._cached_labels[local_idx])

        # HWC uint8 -> CHW float32
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0

        if self.transform:
            image = self.transform(image)

        return image, label


def benchmark_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    num_batches = len(loader)
    print(f"  workers={num_workers}, batch_size={batch_size}, "
          f"batches={num_batches}, samples={len(dataset)}")

    # warmup
    warmup_batches = min(5, num_batches // 10)
    warmup_iter = iter(loader)
    for _ in range(warmup_batches):
        try:
            next(warmup_iter)
        except StopIteration:
            break
    del warmup_iter

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    n_samples = 0

    for images, labels in loader:
        n_samples += images.shape[0]

    elapsed = time.perf_counter() - start_time
    throughput = n_samples / elapsed if elapsed > 0 else 0

    print(f"    -> {elapsed:.2f}s | {throughput:.1f} samples/s")

    result = {
        "num_workers": num_workers,
        "batch_size": batch_size,
        "total_samples": n_samples,
        "num_batches": num_batches,
        "epoch_time_sec": round(elapsed, 4),
        "throughput_samples_per_sec": round(throughput, 2),
    }

    del loader
    return result


def compute_scaling_efficiency(results: List[dict]) -> List[dict]:
    t1 = next((r["epoch_time_sec"] for r in results if r["num_workers"] == 1), None)
    if t1 is None:
        t1 = results[0]["epoch_time_sec"]

    for r in results:
        n, tn = r["num_workers"], r["epoch_time_sec"]
        r["scaling_efficiency_pct"] = round((t1 / (n * tn)) * 100, 2) if n > 0 and tn > 0 else 100.0
        r["speedup"] = round(t1 / tn, 2) if tn > 0 else 0

    return results


def print_results_table(results: List[dict]):
    print()
    print(f"{'Workers':>8} | {'Epoch Time (s)':>15} | {'Throughput (s/s)':>17} | "
          f"{'Speedup':>8} | {'Efficiency (%)':>15}")
    print("-" * 75)
    for r in results:
        print(f"{r['num_workers']:>8} | "
              f"{r['epoch_time_sec']:>15.2f} | "
              f"{r['throughput_samples_per_sec']:>17.1f} | "
              f"{r.get('speedup', '-'):>8} | "
              f"{r.get('scaling_efficiency_pct', '-'):>15}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch DataLoader Microbenchmark")
    parser.add_argument("--data_dir", type=str, default="./synthetic_data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    print(f"data_dir={args.data_dir}, batch_size={args.batch_size}, workers={args.workers}")
    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")

    dataset = SyntheticFileDataset(args.data_dir)
    print(f"loaded {dataset.num_shards} shards x {dataset.samples_per_file} samples = {dataset.total_samples} total\n")

    results = []
    for nw in args.workers:
        print(f"--- workers={nw} ---")
        results.append(benchmark_dataloader(dataset, args.batch_size, nw))

    results = compute_scaling_efficiency(results)
    print_results_table(results)

    output_data = {
        "config": {
            "data_dir": args.data_dir,
            "batch_size": args.batch_size,
            "num_shards": dataset.num_shards,
            "samples_per_file": dataset.samples_per_file,
            "total_samples": dataset.total_samples,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
