# HEP-CCE GSoC '26 | PyTorch DataLoader Microbenchmark & Darshan I/O Analysis

> 📄 **Report:** [View/Download Report (PDF)](./HEPCCE-REPORT-GSOC26-ROHIT_S.pdf)

## Author
**Rohit S**

## Project Structure

```
├── generate_dataset.py      # Phase 1: Synthetic dataset generation
├── benchmark.py             # Phase 2 & 3: DataLoader benchmark
├── darshan_analysis.py      # Darshan I/O analysis script
├── HEPCCE-REPORT-GSOC26-ROHIT_S.pdf     # Final technical report
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── darshan_reports/         # Provided Darshan HTML reports
    ├── 1/run_1_ranks.html
    ├── 2/run_2_ranks.html
    ├── 4/run_4_ranks.html
    ├── 8/run_8_ranks.html
    └── 32/run_32_ranks.html
```

## Environment Setup

### Prerequisites
- Python 3.8+
- around 5-10 GB free disk space (for synthetic dataset)

### Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

## Running the Code

### Step 1: Generate Synthetic Dataset (Phase 1)

```bash
python generate_dataset.py --output_dir ./synthetic_data --total_gb 5 --seed 42
```

This creates around 5 GB of `.npz` shard files in `./synthetic_data/`. Each shard contains 3,500 synthetic 224x224x3 images with labels. Takes around 5-15 minutes depending on disk speed.

**Options:**
- `--output_dir`: Directory for generated data (default: `./synthetic_data`)
- `--total_gb`: Target dataset size in GB (default: `5.0`)
- `--seed`: Random seed for reproducibility (default: `42`)

### Step 2: Run DataLoader Benchmark (Phase 2 & 3)

```bash
python benchmark.py --data_dir ./synthetic_data --batch_size 64 --workers 1 2 4 8
```

This benchmarks the PyTorch DataLoader with `num_workers` = 1, 2, 4, 8. Results are printed as a table and saved to `benchmark_results.json`.

**Options:**
- `--data_dir`: Path to the synthetic dataset (default: `./synthetic_data`)
- `--batch_size`: Batch size (default: `64`)
- `--workers`: List of num_workers to test (default: `1 2 4 8`)
- `--output`: Output JSON file (default: `benchmark_results.json`)

### Step 3: Run Darshan I/O Analysis

```bash
python darshan_analysis.py --output darshan_analysis.json
```

This analyzes the provided Darshan reports and prints a comprehensive comparison of I/O metrics across 1, 2, 4, 8, and 32 thread configurations. No additional software beyond Python is required.

## Output Files

| File | Description |
|------|-------------|
| `benchmark_results.json` | DataLoader benchmark results (timing, throughput, scaling) |
| `darshan_analysis.json` | Extracted Darshan I/O metrics in JSON format |
| `report.pdf` | Final technical report summarizing all findings |

## Reproducing Results

```bash
# Full pipeline (from clean state)
pip install -r requirements.txt
python generate_dataset.py
python benchmark.py
python darshan_analysis.py
```
Run the steps in order from a clean environment. Refer to report.pdf for full analysis and findings.
