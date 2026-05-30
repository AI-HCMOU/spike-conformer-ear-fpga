#!/usr/bin/env python3
"""
Download EarVN1.0 dataset from Mendeley Data.

Usage:
    python scripts/download_data.py --output data/EarVN1.0
"""

import argparse
import os
import zipfile
from pathlib import Path


DATASET_URL = "https://data.mendeley.com/datasets/yws3v3mwx3/4"
DATASET_DOI = "10.17632/yws3v3mwx3.4"


def main():
    parser = argparse.ArgumentParser(description="Download EarVN1.0 dataset")
    parser.add_argument("--output", type=str, default="data/EarVN1.0")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Dataset directory already exists: {output_dir}")
        print("Skipping download. Delete the directory to re-download.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EarVN1.0 Dataset Download Instructions")
    print("=" * 60)
    print()
    print("The EarVN1.0 dataset must be downloaded manually from Mendeley Data:")
    print()
    print(f"  URL: {DATASET_URL}")
    print(f"  DOI: {DATASET_DOI}")
    print()
    print("Dataset details:")
    print("  - 28,412 ear images from 164 subjects")
    print("  - Collected under varying conditions (pose, illumination)")
    print("  - Organized in folders by subject ID")
    print()
    print("After downloading:")
    print(f"  1. Extract the zip to: {output_dir.resolve()}")
    print("  2. Expected structure:")
    print(f"       {output_dir}/001/*.jpg")
    print(f"       {output_dir}/002/*.jpg")
    print(f"       {output_dir}/...")
    print(f"       {output_dir}/164/*.jpg")
    print()

    # Attempt automated download via requests
    try:
        import requests

        print("Attempting automated download...")
        print("Note: Mendeley may require authentication for large datasets.")
        print("If this fails, please download manually from the URL above.")
        print()

        # Mendeley's API endpoint for dataset files
        api_url = "https://data.mendeley.com/api/datasets/yws3v3mwx3/versions/4/files"
        resp = requests.get(api_url, timeout=30)

        if resp.status_code == 200:
            files = resp.json()
            print(f"Found {len(files)} files in dataset.")
            for f in files[:5]:
                print(f"  - {f.get('filename', 'unknown')} ({f.get('size', 0) / 1e6:.1f} MB)")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
            print()
            print("To download all files, visit the URL above in your browser.")
        else:
            print(f"API returned status {resp.status_code}. Please download manually.")

    except ImportError:
        print("Install 'requests' package for automated download: pip install requests")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from the URL above.")


if __name__ == "__main__":
    main()
