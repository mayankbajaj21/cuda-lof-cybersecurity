"""
CICIDS2017 Dataset Acquisition Reference
Milestone M1: Dataset Preparation

Provides official source URLs, file manifests, and download guidance for the
Canadian Institute for Cybersecurity (CIC) CICIDS2017 intrusion detection dataset.
"""

import os
import urllib.request
from typing import Dict

# Official source information
DATASET_SOURCE_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"
OFFICIAL_MIRROR_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/GeneratedLabelledFlows.zip"

RAW_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]


def check_local_dataset(raw_dir: str = "data/raw") -> Dict[str, bool]:
    """Check existence of expected raw CSV files."""
    print("=" * 60)
    print("CICIDS2017 LOCAL DATASET CHECK")
    print("=" * 60)
    print(f"Target raw directory: {raw_dir}")
    print(f"Official Homepage: {DATASET_SOURCE_URL}")
    print(f"Direct Archive: {OFFICIAL_MIRROR_URL}\n")

    status = {}
    total_size = 0
    missing = 0

    for fname in RAW_FILES:
        fpath = os.path.join(raw_dir, fname)
        exists = os.path.exists(fpath)
        status[fname] = exists
        if exists:
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            total_size += size_mb
            print(f"  [FOUND]   {fname:<55} ({size_mb:6.2f} MB)")
        else:
            missing += 1
            print(f"  [MISSING] {fname:<55}")

    print("\n" + "-" * 60)
    if missing == 0:
        print(f"All 8 raw CSV files present locally. Total raw size: {total_size:6.2f} MB")
    else:
        print(f"Warning: {missing} files missing. Please download GeneratedLabelledFlows.zip")
        print(f"from {DATASET_SOURCE_URL} and extract all CSV files into {raw_dir}/")
    print("=" * 60)

    return status


if __name__ == "__main__":
    check_local_dataset()
