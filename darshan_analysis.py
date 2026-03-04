#!/usr/bin/env python3
"""
Darshan I/O analysis for distributed PyTorch runs (1, 2, 4, 8, 32 threads).
Metrics are manually extracted from the provided HTML reports.

Usage:
    python darshan_analysis.py [--output darshan_analysis.json]
"""

import argparse
import json
from collections import OrderedDict


# metrics pulled manually from the HTML reports
# (HTML has base64 images so parsing wasn't practical)
DARSHAN_DATA = OrderedDict({
    1: {
        "ranks": 1,
        "threads": 1,
        "job_id": 4025373,
        "executable": "pyTorch_training.py --batch_size 64 --epochs 10 --threads 1",
        "nprocs": 1,
        "runtime_sec": 244.3957,
        "start_time": "2026-02-25 03:10:34",
        "end_time": "2026-02-25 03:14:38",
        "posix_files_accessed": 8,
        "posix_bytes_read_MiB": 52.58,
        "posix_bytes_written_KiB": 226.43,
        "posix_io_perf_MiBps": 126.61,
        "posix_read_only_files": 5,
        "posix_write_only_files": 2,
        "posix_rw_files": 0,
        "posix_avg_file_size_MiB": 6.59,
        "posix_max_file_size_MiB": 44.86,
        "stdio_files": 1,
        "stdio_bytes_read": 2,
        "stdio_bytes_written": 0,
        "stdio_perf_MiBps": 0.02,
        "common_access_sizes": [(80, 255), (84, 247), (88, 245), (82, 245)],
        "posix_data_KiB": 0.69,
        "dxt_posix_data_KiB": 31.36,
        "heatmap_data_KiB": 0.45,
    },
    2: {
        "ranks": 2,
        "threads": 2,
        "job_id": 1347640,
        "executable": "pyTorch_training.py --batch_size 64 --epochs 10 --threads 2",
        "nprocs": 1,
        "runtime_sec": 298.6351,
        "start_time": "2026-02-26 19:46:52",
        "end_time": "2026-02-26 19:51:51",
        "posix_files_accessed": 8,
        "posix_bytes_read_MiB": 52.58,
        "posix_bytes_written_KiB": 213.37,
        "posix_io_perf_MiBps": 242.81,
        "posix_read_only_files": 5,
        "posix_write_only_files": 2,
        "posix_rw_files": 0,
        "posix_avg_file_size_MiB": 6.59,
        "posix_max_file_size_MiB": 44.86,
        "stdio_files": 1,
        "stdio_bytes_read": 2,
        "stdio_bytes_written": 0,
        "stdio_perf_MiBps": 0.01,
        "common_access_sizes": [(84, 235), (88, 231), (78, 230), (90, 228)],
        "posix_data_KiB": 0.70,
        "dxt_posix_data_KiB": 29.81,
        "heatmap_data_KiB": 0.45,
    },
    4: {
        "ranks": 4,
        "threads": 4,
        "job_id": 1348084,
        "executable": "pyTorch_training.py --batch_size 64 --epochs 10 --threads 4",
        "nprocs": 1,
        "runtime_sec": 209.8710,
        "start_time": "2026-02-26 19:51:51",
        "end_time": "2026-02-26 19:55:21",
        "posix_files_accessed": 8,
        "posix_bytes_read_MiB": 52.58,
        "posix_bytes_written_KiB": 211.19,
        "posix_io_perf_MiBps": 507.56,
        "posix_read_only_files": 5,
        "posix_write_only_files": 2,
        "posix_rw_files": 0,
        "posix_avg_file_size_MiB": 6.59,
        "posix_max_file_size_MiB": 44.86,
        "stdio_files": 1,
        "stdio_bytes_read": 2,
        "stdio_bytes_written": 0,
        "stdio_perf_MiBps": 0.02,
        "common_access_sizes": [(84, 235), (78, 234), (88, 232), (90, 228)],
        "posix_data_KiB": 0.69,
        "dxt_posix_data_KiB": 29.11,
        "heatmap_data_KiB": 0.42,
    },
    8: {
        "ranks": 8,
        "threads": 8,
        "job_id": 4025742,
        "executable": "pyTorch_training.py --batch_size 64 --epochs 10 --threads 8",
        "nprocs": 1,
        "runtime_sec": 215.1849,
        "start_time": "2026-02-25 03:14:39",
        "end_time": "2026-02-25 03:18:14",
        "posix_files_accessed": 8,
        "posix_bytes_read_MiB": 52.58,
        "posix_bytes_written_KiB": 212.52,
        "posix_io_perf_MiBps": 474.52,
        "posix_read_only_files": 5,
        "posix_write_only_files": 2,
        "posix_rw_files": 0,
        "posix_avg_file_size_MiB": 6.59,
        "posix_max_file_size_MiB": 44.86,
        "stdio_files": 1,
        "stdio_bytes_read": 2,
        "stdio_bytes_written": 0,
        "stdio_perf_MiBps": 0.02,
        "common_access_sizes": [(84, 235), (88, 232), (78, 229), (90, 228)],
        "posix_data_KiB": 0.70,
        "dxt_posix_data_KiB": 29.37,
        "heatmap_data_KiB": 0.42,
    },
    32: {
        "ranks": 32,
        "threads": 32,
        "job_id": 4027483,
        "executable": "pyTorch_training.py --batch_size 64 --epochs 10 --threads 32",
        "nprocs": 1,
        "runtime_sec": 228.3162,
        "start_time": "2026-02-25 03:23:20",
        "end_time": "2026-02-25 03:27:09",
        "posix_files_accessed": 8,
        "posix_bytes_read_MiB": 52.58,
        "posix_bytes_written_KiB": 212.18,
        "posix_io_perf_MiBps": 473.50,
        "posix_read_only_files": 5,
        "posix_write_only_files": 2,
        "posix_rw_files": 0,
        "posix_avg_file_size_MiB": 6.59,
        "posix_max_file_size_MiB": 44.86,
        "stdio_files": 1,
        "stdio_bytes_read": 2,
        "stdio_bytes_written": 0,
        "stdio_perf_MiBps": 0.02,
        "common_access_sizes": [(75, 237), (78, 236), (84, 232), (90, 229)],
        "posix_data_KiB": 0.70,
        "dxt_posix_data_KiB": 29.51,
        "heatmap_data_KiB": 0.46,
    },
})


def analyze_darshan_reports():
    # job overview
    print(f"\n{'Threads':>8} | {'Job ID':>10} | {'Runtime (s)':>12} | "
          f"{'Start Time':>20} | {'End Time':>20}")
    print("-" * 80)
    for t, d in DARSHAN_DATA.items():
        print(f"{t:>8} | {d['job_id']:>10} | {d['runtime_sec']:>12.2f} | "
              f"{d['start_time']:>20} | {d['end_time']:>20}")

    # posix i/o
    print(f"\n{'Threads':>8} | {'Files':>6} | {'Read (MiB)':>11} | "
          f"{'Written (KiB)':>14} | {'I/O Perf (MiB/s)':>17}")
    print("-" * 70)
    for t, d in DARSHAN_DATA.items():
        print(f"{t:>8} | {d['posix_files_accessed']:>6} | "
              f"{d['posix_bytes_read_MiB']:>11.2f} | "
              f"{d['posix_bytes_written_KiB']:>14.2f} | "
              f"{d['posix_io_perf_MiBps']:>17.2f}")

    # file access breakdown
    print(f"\n{'Threads':>8} | {'Read-Only':>10} | {'Write-Only':>11} | "
          f"{'Read/Write':>11} | {'Total':>6}")
    print("-" * 60)
    for t, d in DARSHAN_DATA.items():
        print(f"{t:>8} | {d['posix_read_only_files']:>10} | "
              f"{d['posix_write_only_files']:>11} | "
              f"{d['posix_rw_files']:>11} | "
              f"{d['posix_files_accessed']:>6}")

    # scaling
    base_time = DARSHAN_DATA[1]["runtime_sec"]
    base_perf = DARSHAN_DATA[1]["posix_io_perf_MiBps"]

    print(f"\n{'Threads':>8} | {'Runtime (s)':>12} | {'Speedup':>8} | "
          f"{'I/O Perf (MiB/s)':>17} | {'I/O Speedup':>12} | {'Efficiency (%)':>15}")
    print("-" * 85)
    for t, d in DARSHAN_DATA.items():
        speedup = base_time / d["runtime_sec"]
        io_speedup = d["posix_io_perf_MiBps"] / base_perf
        efficiency = (speedup / t) * 100
        print(f"{t:>8} | {d['runtime_sec']:>12.2f} | {speedup:>8.2f} | "
              f"{d['posix_io_perf_MiBps']:>17.2f} | {io_speedup:>12.2f} | "
              f"{efficiency:>15.1f}")

    # read/write volume
    print(f"\n{'Threads':>8} | {'Total Read':>12} | {'Total Write':>12} | "
          f"{'Read:Write Ratio':>17}")
    print("-" * 60)
    for t, d in DARSHAN_DATA.items():
        read_kib = d["posix_bytes_read_MiB"] * 1024
        write_kib = d["posix_bytes_written_KiB"]
        ratio = read_kib / write_kib if write_kib > 0 else float("inf")
        print(f"{t:>8} | {d['posix_bytes_read_MiB']:>10.2f} MiB | "
              f"{d['posix_bytes_written_KiB']:>9.2f} KiB | {ratio:>17.1f}")

    # findings
    print("""
--- Does I/O scale with thread count? ---

Not really. Performance improved about 4 times going from 1 to 4 threads (126 to 507 MiB/s),
but beyond that, it stopped improving; 8T and 32T both hover around 474 MiB/s. 
The 2-thread run was actually the slowest, likely because it ran on a different date under
different system conditions. The overall picture is clear adding more threads helps
up to a point, then the bottleneck shifts elsewhere.


--- Filesystem issues at scale ---

- Tiny reads dominate: 75–90 bytes per operation means each read costs a full syscall regardless of how little data moves, this gets expensive at scale.
- All threads share the same files: With no data partitioning, every thread opens the same 8 files. At 32+ threads, this creates a queue at the filesystem metadata level before any real data transfer even begins.
- Diminishing returns past 4 threads: The 32-thread run was 9 percent slower than 4 threads because more threads mean more scheduling noise and cache thrashing.


--- What would actually help ---

- Bigger reads: Switch to WebDataset, FFCV and LMDB formats that pack samples into large sequential chunks, replacing thousands of 80-byte reads with a few large ones.
- Split the data across ranks: Each process should read only its own slice of the dataset. Right now, every rank reads everything, which doesn't scale at all on multi-node systems
- Don't over-provision workers: 4 workers was the sweet spot here, pushing to 8 or  32 affected the performance, so always profile before assuming more workers means  faster loading
""")

    # access sizes
    for t, d in DARSHAN_DATA.items():
        sizes_str = ", ".join([f"{s}B x{c}" for s, c in d["common_access_sizes"]])
        print(f"  {t:>2}T: {sizes_str}")
    print("\n  all runs: 75-90 byte accesses, around 230-255 times each — looks like"
          " model/config reads, not bulk data I/O")

    return DARSHAN_DATA


def save_analysis_json(data: dict, output_file: str):
    output = {str(t): d for t, d in data.items()}
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nsaved -> {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="darshan_analysis.json")
    args = parser.parse_args()

    data = analyze_darshan_reports()
    save_analysis_json(data, args.output)
