"""
STREEVA — Area Risk Scoring Engine
scripts/ingest_ncrb.py

NCRB District-level IPC Crime Data Ingestion Pipeline.

Data source:
  The bundled file data/raw/ncrb_ipc_crimes_2014.json contains district-wise
  IPC crime counts for all Indian districts (2014), sourced from data.gov.in.

  Original dataset: "District Wise Crime Data" from NCRB / data.gov.in
  Format: JSON with 'fields' schema and 'data' array (87 crime columns)

What this script does:
  1. Loads data/raw/ncrb_ipc_crimes_2014.json
  2. Extracts Tamil Nadu districts
  3. Computes a "violent crime" count per district (curated subset of IPC fields)
  4. Normalises violent crime rate across TN districts to a 0–100 risk score
  5. Writes data/processed/ncrb_district_lookup.json

Why violent crimes only (not total IPC)?
  Total IPC for Chennai = 16,861
  Of which: Rash Driving alone = 8,527 (a traffic offence, not violence)
  Using total IPC would conflate traffic safety with personal safety.
  We select offences that are directly relevant to personal safety risk:
  murder, rape, kidnapping, robbery, assault, dacoity, extortion, etc.

How to update when NCRB publishes a new annual report:
  1. Download the new district-wise CSV/JSON from data.gov.in
  2. Replace data/raw/ncrb_ipc_crimes_2014.json (or add a new year file)
  3. Update YEAR constant below
  4. Run: python scripts/ingest_ncrb.py
  5. Restart the service

Run from project root:
    python scripts/ingest_ncrb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
INPUT_FILE = ROOT / "data" / "raw" / "ncrb_ipc_crimes_2014.json"
OUTPUT_FILE = ROOT / "data" / "processed" / "ncrb_district_lookup.json"
YEAR = "2014"

# ── Violent crime field labels ─────────────────────────────────────────────────
# These are the fields from the dataset that directly relate to personal safety.
# Rash driving, counterfeiting, breach of trust are excluded.
VIOLENT_CRIME_FIELDS = {
    "Murder",
    "Attempt to commit Murder",
    "Culpable Homicide not amounting to Murder",
    "Rape",
    "Attempt to commit Rape",
    "Kidnapping & Abduction_Total",
    "Dacoity",
    "Dacoity with Murder",
    "Robbery",
    "Arson",
    "Grievous Hurt",
    "Hurt",
    "Acid attack",
    "Attempt to Acid Attack",
    "Dowry Deaths",
    "Assault on Women with intent to outrage her Modesty",
    "Sexual Harassment",
    "Stalking",
    "Voyeurism",
    "Extortion",
    "HumanTrafficking",
    "Unnatural Offence",
}


def load_dataset(path: Path) -> tuple[list[dict], list[list]]:
    """Load the NCRB JSON dataset and return (fields, data)."""
    print(f"Loading: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    fields = raw["fields"]
    data = raw["data"]
    print(f"  Fields: {len(fields)}, Rows: {len(data)}")
    return fields, data


def build_field_index(fields: list[dict]) -> dict[str, int]:
    """Build a mapping of {field_label: column_index}."""
    return {field["label"]: idx for idx, field in enumerate(fields)}


def compute_violent_crimes(row: list, field_index: dict[str, int]) -> int:
    """Sum up violent crime counts for a single district row."""
    total = 0
    for label, idx in field_index.items():
        if label in VIOLENT_CRIME_FIELDS:
            try:
                total += int(row[idx] or 0)
            except (ValueError, TypeError):
                pass
    return total


def ingest(state_filter: str = "Tamil Nadu") -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        state_filter: Only process districts from this state.

    Returns:
        Dict mapping district name → crime stats.
    """
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Place the NCRB dataset JSON at data/raw/ncrb_ipc_crimes_2014.json")
        sys.exit(1)

    fields, data = load_dataset(INPUT_FILE)
    field_index = build_field_index(fields)

    # Column indices for key fields
    state_col = field_index.get("States/UTs", 0)
    district_col = field_index.get("District", 1)
    year_col = field_index.get("Year", 2)
    total_ipc_col = field_index.get("Total Cognizable IPC crimes", len(fields) - 1)

    # Filter to target state
    state_rows = [
        row for row in data
        if str(row[state_col]).strip() == state_filter
           and str(row[district_col]).strip() != "Total"
           and str(row[district_col]).strip() != "Other Units"
           and str(row[district_col]).strip() != "Cyber Cell"
    ]

    print(f"\nProcessing {state_filter} districts: {len(state_rows)} rows")

    # Compute violent crimes per district
    district_stats = []
    for row in state_rows:
        district = str(row[district_col]).strip()
        violent = compute_violent_crimes(row, field_index)
        try:
            total_ipc = int(row[total_ipc_col] or 0)
        except (ValueError, TypeError):
            total_ipc = 0

        district_stats.append({
            "district": district,
            "year": str(row[year_col]).strip(),
            "violent_crimes": violent,
            "total_ipc_crimes": total_ipc,
        })
        print(f"  {district}: violent={violent}, total_ipc={total_ipc}")

    # Normalise violent crimes to 0–100 risk score across TN districts
    violent_counts = [d["violent_crimes"] for d in district_stats]
    v_min = min(violent_counts)
    v_max = max(violent_counts)

    print(f"\nNormalization: min={v_min}, max={v_max}")

    # Sort by violent crimes descending for state rank
    district_stats.sort(key=lambda x: x["violent_crimes"], reverse=True)

    lookup = {}
    for rank, stats in enumerate(district_stats, start=1):
        if v_max > v_min:
            normalized = (stats["violent_crimes"] - v_min) / (v_max - v_min) * 100
        else:
            normalized = 50.0

        lookup[stats["district"]] = {
            "violent_crimes": stats["violent_crimes"],
            "total_ipc_crimes": stats["total_ipc_crimes"],
            "year": stats["year"],
            "normalized_score": round(normalized, 2),
            "state_rank": rank,
            "state_district_count": len(district_stats),
            "normalization": {
                "method": "min-max across Tamil Nadu districts",
                "min_violent": v_min,
                "max_violent": v_max,
                "formula": "(violent_crimes - min) / (max - min) × 100",
            },
        }

    # Print Chennai specifically
    if "Chennai" in lookup:
        chennai = lookup["Chennai"]
        print(f"\nChennai: violent_crimes={chennai['violent_crimes']}, "
              f"normalized_score={chennai['normalized_score']:.1f}, "
              f"state_rank={chennai['state_rank']}/{chennai['state_district_count']}")

    return lookup


def save_lookup(lookup: dict) -> None:
    """Write the processed lookup to data/processed/ncrb_district_lookup.json."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Districts in lookup: {len(lookup)}")


if __name__ == "__main__":
    print("=" * 60)
    print("STREEVA — NCRB Crime Data Ingestion")
    print(f"Year: {YEAR}")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)

    lookup = ingest(state_filter="Tamil Nadu")
    save_lookup(lookup)

    print("\nDone. The MacroBaselineService will load this file at startup.")
    print("Re-run this script when NCRB publishes a new annual report.")
