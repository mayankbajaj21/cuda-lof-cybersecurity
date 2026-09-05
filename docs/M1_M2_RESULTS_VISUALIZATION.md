# Milestone M1 & M2: Results Visualization & Tabular Report

## 1. Overview & Purpose
This document consolidates all empirical tabular findings and research-grade visual charts generated from the actual execution outputs of:
- **Milestone M1 — Dataset Preparation** (CICIDS2017 ingestion, cleaning, feature selection, and subset creation).
- **Milestone M2 — Naive CPU LOF Baseline** (First-principles Local Outlier Factor implementation, verification, and empirical scaling).

All charts and tables presented herein are derived exclusively from authentic project data files without artificial or extrapolated values.

---

## 2. Directory Structure of Results
 
```text
results/
├── m1_dataset_preparation/
│   ├── tables/
│   │   ├── m1_dataset_summary.csv        # Table 1: Ingestion, cleaning & retention metrics
│   │   ├── m1_subset_summary.csv         # Table 2: Subset configurations (10K to 1M)
│   │   └── m1_class_distribution.csv     # Table 3: 15-class distribution (Benign & Attacks)
│   └── charts/
│       ├── m1_dataset_reduction.png      # Chart 1: Cleansing & reduction breakdown
│       ├── m1_class_distribution.png     # Chart 2: Full horizontal class distribution (log scale)
│       └── m1_processed_subsets.png      # Chart 3: Stratified subset scale breakdown
│
└── m2_cpu_baseline/
    ├── benchmarks/
    │   ├── cpu_lof_10k_k20.json          # Raw JSON output for 10K run
    │   ├── cpu_lof_25k_k20.json          # Raw JSON output for 25K run
    │   ├── lof_scores_10k_k20.npy        # Raw score vector (10K)
    │   └── lof_scores_25k_k20.npy        # Raw score vector (25K)
    ├── tables/
    │   ├── m2_cpu_benchmark.csv          # Table 4: End-to-end CPU LOF benchmarks (10K & 25K)
    │   ├── m2_lof_statistics.csv         # Table 5: Statistical score distribution
    │   └── m2_runtime_breakdown.csv      # Table 6: Fine-grained stage runtime breakdown & %
    └── charts/
        ├── m2_cpu_execution_time.png     # Chart 4: 10K vs 25K compute time comparison
        ├── m2_runtime_breakdown_25k.png  # Chart 5: Stage runtime % breakdown (25K bottleneck)
        ├── m2_cpu_scaling.png            # Chart 6: Measured empirical CPU scaling curve
        └── m2_pairwise_vs_knn.png        # Chart 7: Distance matrix vs. k-NN search comparison
```

---

## 3. Milestone M1: Dataset Preparation Tables & Charts

### 3.1 Data Provenance & Source
- **Source**: `data/processed/dataset_metadata.json`
- **Script**: `src/python/visualization/m1_visualizations.py`

### 3.2 Tabular Outputs

#### Table 1: Dataset Preparation Summary (`m1_dataset_summary.csv`)
| Metric | Result |
| :--- | :--- |
| **Raw CSV Files** | 8 |
| **Raw Samples** | 2,830,743 |
| **Raw Features** | 79 |
| **Numerical Features (Raw)** | 78 |
| **Non-Numerical Features (Raw)** | 1 (`Label`) |
| **NaN Rows Removed** | 1,358 |
| **Inf Rows Removed** | 2,867 |
| **Total Invalid Rows Removed** | 2,867 |
| **Duplicate Rows Removed** | 307,078 |
| **Cleaned Samples** | 2,520,798 |
| **Retention Rate** | 89.05% |
| **Final Numerical Features** | 68 |
| **Excluded Features Total** | 11 |
| **Removed Zero-Variance Features** | 8 |
| **Removed Identifiers / Redundant** | 3 (`Label`, `Destination Port`, `Fwd Header Length.1`) |
| **Final Benign Samples** | 2,095,057 (83.1109%) |
| **Final Anomaly Samples** | 425,741 (16.89%) |
| **Scaling Method** | `StandardScaler` (z-score normalization) |
| **Data Type** | `float32` (IEEE 754 single precision) |
| **Random Seed** | 42 |

#### Table 2: Processed Dataset Subsets (`m1_subset_summary.csv`)
| Subset | Samples | File | File Size (MB) | Benign Count | Anomaly Count | Anomaly % |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **10K** | 10,000 | `subset_10k.npz` | 0.93 | 8,308 | 1,692 | 16.92% |
| **25K** | 25,000 | `subset_25k.npz` | 2.28 | 20,776 | 4,224 | 16.90% |
| **100K** | 100,000 | `subset_100k.npz` | 8.94 | 83,111 | 16,889 | 16.89% |
| **250K** | 250,000 | `subset_250k.npz` | 21.97 | 207,777 | 42,223 | 16.89% |
| **500K** | 500,000 | `subset_500k.npz` | 43.19 | 415,555 | 84,445 | 16.89% |
| **1M** | 1,000,000 | `subset_1m.npz` | 84.60 | 831,110 | 168,890 | 16.89% |
| **Full** | 2,520,798 | `cicids2017_clean_full.npz` | 173.19 | 2,095,057 | 425,741 | 16.89% |

#### Table 3: Final Class Distribution (`m1_class_distribution.csv`)
| Class ID | Class Name | Category | Samples | Percentage |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `BENIGN` | Normal | 2,095,057 | 83.1109% |
| 4 | `DoS Hulk` | Attack | 172,846 | 6.8568% |
| 2 | `DDoS` | Attack | 128,014 | 5.0783% |
| 10 | `PortScan` | Attack | 90,694 | 3.5978% |
| 3 | `DoS GoldenEye` | Attack | 10,286 | 0.4080% |
| 7 | `FTP-Patator` | Attack | 5,931 | 0.2353% |
| 6 | `DoS slowloris` | Attack | 5,385 | 0.2136% |
| 5 | `DoS Slowhttptest` | Attack | 5,228 | 0.2074% |
| 11 | `SSH-Patator` | Attack | 3,219 | 0.1277% |
| 1 | `Bot` | Attack | 1,948 | 0.0773% |
| 12 | `Web Attack - Brute Force` | Attack | 1,470 | 0.0583% |
| 14 | `Web Attack - XSS` | Attack | 652 | 0.0259% |
| 9 | `Infiltration` | Attack | 36 | 0.0014% |
| 13 | `Web Attack - Sql Injection` | Attack | 21 | 0.0008% |
| 8 | `Heartbleed` | Attack | 11 | 0.0004% |

### 3.3 Visual Charts Description

- **Chart 1 (`m1_dataset_reduction.png`)**:
  - *Demonstrates*: Quantifies the impact of data cleansing. Shows that invalid records (2,867) and redundant duplicates (307,078) represent 10.95% of raw captures, leaving a high-fidelity benchmark corpus of 2,520,798 valid records.
- **Chart 2 (`m1_class_distribution.png`)**:
  - *Demonstrates*: Log-scaled horizontal bar chart detailing the severe class imbalance characteristic of real-world cybersecurity traffic (from 2.09M benign flows down to 11 Heartbleed attacks). Confirms why unsupervised density estimation (LOF) is well-suited for zero-day threat detection.
- **Chart 3 (`m1_processed_subsets.png`)**:
  - *Demonstrates*: The structural progression of controlled subsets (10K to 1M) created for GPU benchmarking, highlighting sample counts and compressed storage efficiency (~0.93 MB to ~84.6 MB).

---

## 4. Milestone M2: CPU Baseline Tables & Charts

### 4.1 Data Provenance & Source
- **Source**: `results/m2_cpu_baseline/benchmarks/cpu_lof_10k_k20.json`, `results/m2_cpu_baseline/benchmarks/cpu_lof_25k_k20.json`
- **Script**: `src/python/visualization/m2_visualizations.py`

### 4.2 Tabular Outputs

#### Table 4: CPU LOF Benchmark Comparison (`m2_cpu_benchmark.csv`)
| Metric | 10K Subset | 25K Subset |
| :--- | :---: | :---: |
| **Samples ($N$)** | 10,000 | 25,000 |
| **Features ($D$)** | 68 | 68 |
| **Neighborhood Parameter ($k$)** | 20 | 20 |
| **Contamination Parameter** | 0.10 | 0.10 |
| **Data Loading Time** | 0.01756 s | 0.03336 s |
| **Pairwise Distance Matrix Time** | 1.39061 s | 10.97823 s |
| **$k$-NN Search & Sort Time** | 1.66738 s | 11.12575 s |
| **Reachability Distance Time** | 0.00417 s | 0.01262 s |
| **LRD Computation Time** | 0.00028 s | 0.01647 s |
| **LOF Scoring Time** | 0.00134 s | 0.00304 s |
| **Total LOF Compute Time** | **3.06379 s** | **22.13613 s** |
| **Total End-to-End Time** | **3.08135 s** | **22.16949 s** |

#### Table 5: LOF Score Statistics (`m2_lof_statistics.csv`)
| Metric | 10K Subset | 25K Subset |
| :--- | :---: | :---: |
| **Minimum LOF Score** | 0.91441 | 0.45000 |
| **Maximum LOF Score** | 1,061.43396 | 2,966,723,072.00000 |
| **Mean LOF Score** | 1.71634 | 1,209,706.75000 |
| **Standard Deviation** | 11.32839 | 49,812,688.00000 |
| **Median ($p_{50}$)** | 1.05598 | 1.05294 |
| **25th Percentile ($p_{25}$)** | 0.99931 | 0.99933 |
| **75th Percentile ($p_{75}$)** | 1.28472 | 1.26669 |
| **90th Percentile ($p_{90}$)** | 1.90915 | 1.94397 |
| **95th Percentile ($p_{95}$)** | 3.19238 | 3.58155 |
| **99th Percentile ($p_{99}$)** | 9.90076 | 23.26931 |
| **Decision Threshold ($\tau_{0.10}$)** | 1.90915 | 1.94397 |
| **Predicted Inliers** | 9,000 | 22,500 |
| **Predicted Anomalies** | 1,000 | 2,500 |

#### Table 6: CPU LOF Runtime Breakdown (`m2_runtime_breakdown.csv`)
| Stage | 10K Time (s) | 10K % | 25K Time (s) | 25K % |
| :--- | :---: | :---: | :---: | :---: |
| **Pairwise Distance Matrix** | 1.39061 s | 45.39% | 10.97823 s | 49.59% |
| **$k$-NN Search & Sort** | 1.66738 s | 54.42% | 11.12575 s | 50.26% |
| **Reachability Distance** | 0.00417 s | 0.14% | 0.01262 s | 0.06% |
| **LRD Computation** | 0.00028 s | 0.01% | 0.01647 s | 0.07% |
| **LOF Scoring** | 0.00134 s | 0.04% | 0.00304 s | 0.01% |
| **Total LOF Compute** | **3.06379 s** | **100.00%** | **22.13613 s** | **100.00%** |

### 4.3 Visual Charts Description

- **Chart 4 (`m2_cpu_execution_time.png`)**:
  - *Demonstrates*: Direct comparison of CPU LOF computation times: 3.064 s on 10K vs 22.136 s on 25K. Highlights that a $2.5\times$ increase in dataset scale causes a **$7.22\times$ runtime increase** on CPU.
- **Chart 5 (`m2_runtime_breakdown_25k.png`)**:
  - *Demonstrates*: Proves the core research motivation for GPU acceleration. On 25K samples, **99.85%** of CPU compute time is consumed exclusively by two stages: Pairwise Distance Matrix (49.59%) and $k$-NN Sorting (50.26%). Density and LOF scoring represent only 0.15%.
- **Chart 6 (`m2_cpu_scaling.png`)**:
  - *Demonstrates*: Empirical scaling curve showing steep quadratic growth ($O(N^2)$) between measured baseline data points ($N=10\text{K}$ and $N=25\text{K}$).
- **Chart 7 (`m2_pairwise_vs_knn.png`)**:
  - *Demonstrates*: Direct stage-by-stage comparison showing that pairwise distance and neighbor sorting scale concurrently as $O(N^2 \cdot D)$ and $O(N^2 \log N)$, forming the dual targets for CUDA kernel parallelization in Milestone M3.

---

## 5. Script Reproducibility

To re-generate all M1 and M2 tables and figures from project data:

```powershell
# 1. Generate Milestone M1 Tables and Charts
python src/python/visualization/m1_visualizations.py

# 2. Generate Milestone M2 Tables and Charts
python src/python/visualization/m2_visualizations.py

# 3. Execute Automated Test Suite
pytest -v tests/
```
