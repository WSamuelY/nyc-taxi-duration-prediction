# ============================================================
# src/01_download_data.py
# Phase 1: Data Acquisition
#
# PURPOSE:
#   Download all 14 months of NYC Yellow Taxi trip data:
#     - January 2025 through December 2025  (12 files)
#     - January 2026 and February 2026      (2 files)
#   Also download the Taxi Zone Lookup Table.
#
# HOW TO RUN:
#   python src/01_download_data.py
#
# OUTPUT:
#   data/raw/yellow_tripdata_YYYY-MM.parquet  (14 files)
#   data/raw/taxi_zone_lookup.csv
# ============================================================

import os
import sys
import time
import requests
from tqdm import tqdm

# Add project root to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, get_data_path, print_section

# ── Logger ───────────────────────────────────────────────────
logger = get_logger("01_download_data")

# ── Constants ────────────────────────────────────────────────

# Base URL for TLC parquet files
# Pattern: yellow_tripdata_YYYY-MM.parquet
TLC_BASE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_{year}-{month:02d}.parquet"
)

# Taxi Zone Lookup Table — maps LocationID to borough/zone name
ZONE_LOOKUP_URL = (
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
)

# All months we need to download
DOWNLOAD_TARGETS = (
    # (year, month) tuples
    [(2025, m) for m in range(1, 13)] +   # Jan–Dec 2025
    [(2026, 1), (2026, 2)]                 # Jan–Feb 2026
)

# ── Download Helper ───────────────────────────────────────────

def download_file(url: str, dest_path: str, desc: str = "") -> bool:
    """
    Download a file from a URL to a local destination path.
    Shows a progress bar and skips files that already exist.

    Args:
        url:       Full URL of the file to download
        dest_path: Local file path to save to
        desc:      Label to show in the progress bar

    Returns:
        True if download succeeded (or was skipped), False on error
    """

    # Skip if already downloaded — saves time on re-runs
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / 1e6
        logger.info(f"  ✓ Already exists ({size_mb:.1f} MB): {os.path.basename(dest_path)}")
        return True

    logger.info(f"  ⬇ Downloading: {desc}")

    try:
        # Stream the download so we can show a progress bar
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()  # Raise error for 4xx / 5xx HTTP codes

        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"  {desc}",
                ncols=80
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        size_mb = os.path.getsize(dest_path) / 1e6
        logger.info(f"  ✓ Saved ({size_mb:.1f} MB): {os.path.basename(dest_path)}")
        return True

    except requests.exceptions.HTTPError as e:
        logger.warning(f"  ✗ HTTP error for {desc}: {e}")
        # Remove partial file if it exists
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

    except Exception as e:
        logger.error(f"  ✗ Unexpected error for {desc}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


# ── Main Download Routine ─────────────────────────────────────

def download_trip_data():
    """
    Download all 14 monthly parquet files to data/raw/.

    The TLC data is partitioned by month. Each file contains all
    yellow taxi trips recorded in that month.
    """
    raw_dir = get_data_path("raw")
    os.makedirs(raw_dir, exist_ok=True)

    print_section("Phase 1: Downloading NYC Yellow Taxi Trip Data")
    logger.info(f"Target directory: {raw_dir}")
    logger.info(f"Files to download: {len(DOWNLOAD_TARGETS)}")

    success_count = 0
    fail_count = 0
    failed_files = []

    for year, month in DOWNLOAD_TARGETS:
        url = TLC_BASE_URL.format(year=year, month=month)
        filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
        dest_path = os.path.join(raw_dir, filename)
        desc = f"{year}-{month:02d}"

        ok = download_file(url, dest_path, desc)

        if ok:
            success_count += 1
        else:
            fail_count += 1
            failed_files.append(desc)

        # Small pause between requests — be polite to the server
        time.sleep(0.5)

    # ── Summary ──────────────────────────────────────────────
    print_section("Download Summary")
    logger.info(f"  ✓ Successful : {success_count} / {len(DOWNLOAD_TARGETS)} files")
    if failed_files:
        logger.warning(f"  ✗ Failed     : {fail_count} files → {failed_files}")
        logger.warning(
            "  NOTE: TLC sometimes delays publishing recent months. "
            "Check https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page "
            "for the latest available months."
        )
    else:
        logger.info("  All files downloaded successfully!")


def download_zone_lookup():
    """
    Download the Taxi Zone Lookup Table.

    This CSV maps each LocationID (1–265) to:
      - Borough       (Manhattan, Brooklyn, Queens, Bronx, Staten Island, EWR)
      - Zone          (e.g., 'JFK Airport', 'Times Sq/Theatre District')
      - service_zone  (Yellow Zone, Boro Zone, Airports, etc.)

    We use this to enrich trip records with human-readable zone names
    and optionally to compute zone-level spatial features.
    """
    raw_dir = get_data_path("raw")
    dest_path = os.path.join(raw_dir, "taxi_zone_lookup.csv")

    print_section("Downloading Taxi Zone Lookup Table")
    download_file(ZONE_LOOKUP_URL, dest_path, "taxi_zone_lookup.csv")


# ── Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NYC YELLOW TAXI — PHASE 1: DATA ACQUISITION")
    print("="*60)

    # Step 1: Download monthly trip parquet files
    download_trip_data()

    # Step 2: Download the zone lookup table
    download_zone_lookup()

    print("\n✅ Phase 1 complete. Check data/raw/ for all downloaded files.")
    print("   Next step: python src/02_prepare_data.py\n")
