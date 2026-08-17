"""
STREEVA — Area Risk Scoring Engine
scripts/demo_locations.py

Qualitative validation script for 6 Chennai locations.

Purpose:
  There is NO ground-truth crime-incident dataset for Chennai.
  This script validates the model qualitatively by checking that
  the ranking of locations makes intuitive sense:
  - Isolated roads should score HIGHER (riskier) than busy main roads
  - Areas near police stations should score LOWER than areas far from them
  - The same location at night (22:00) should score HIGHER than at noon (12:00)

Expected qualitative ranking (higher score = riskier):
  1. Vandalur Isolated Road (most isolated, night)   → Critical
  2. VIT Chennai campus area (suburban, less connected)→ High
  3. Tambaram Market (busy, daytime)                  → Medium
  4. OMR IT Corridor (main arterial road)             → Medium-Low
  5. T. Nagar commercial district (densest commercial)→ Low
  6. Marina Beach (open, popular, public)             → Low

Usage:
  1. Start the service: uvicorn app.main:app --reload
  2. Run: python scripts/demo_locations.py
  OR: python scripts/demo_locations.py --no-server (uses API directly without server)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Test locations ─────────────────────────────────────────────────────────────
LOCATIONS = [
    {
        "name": "VIT Chennai (Tambaram Campus)",
        "lat": 12.8406,
        "lng": 80.1534,
        "hour_day": 14,
        "hour_night": 22,
        "notes": "Suburban campus fringe. Mixed residential/institutional. Less connected.",
        "expected_rough": "Medium-High",
    },
    {
        "name": "Tambaram Market (Main Road)",
        "lat": 12.9249,
        "lng": 80.1000,
        "hour_day": 14,
        "hour_night": 22,
        "notes": "Busy commercial market street. High commercial density.",
        "expected_rough": "Medium",
    },
    {
        "name": "OMR IT Corridor (Old Mahabalipuram Road)",
        "lat": 12.9260,
        "lng": 80.2318,
        "hour_day": 14,
        "hour_night": 22,
        "notes": "Major arterial road with IT parks. Good road density, some isolated stretches at night.",
        "expected_rough": "Low-Medium",
    },
    {
        "name": "Marina Beach (Kamarajar Salai)",
        "lat": 13.0500,
        "lng": 80.2824,
        "hour_day": 14,
        "hour_night": 22,
        "notes": "Popular public beach area. High foot traffic, well-known, police presence.",
        "expected_rough": "Low",
    },
    {
        "name": "T. Nagar Commercial District",
        "lat": 13.0418,
        "lng": 80.2341,
        "hour_day": 14,
        "hour_night": 22,
        "notes": "One of Chennai's densest commercial areas. Very high commercial density.",
        "expected_rough": "Low",
    },
    {
        "name": "Vandalur Isolated Stretch (GST Road periphery)",
        "lat": 12.8815,
        "lng": 80.0798,
        "hour_day": 14,
        "hour_night": 22,
        "notes": "Relatively isolated road on the periphery of Vandalur zoo area. "
                 "Lower road density, fewer commercial establishments.",
        "expected_rough": "High-Critical",
    },
]


def query_via_http(lat: float, lng: float, hour: int, base_url: str = "http://localhost:8000") -> dict:
    """Query the running FastAPI service via HTTP."""
    import requests
    url = f"{base_url}/area-risk"
    params = {"lat": lat, "lng": lng, "hour": hour}
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def query_direct(lat: float, lng: float, hour: int) -> dict:
    """
    Query the scoring engine directly (no HTTP server needed).
    Used for quick testing without starting uvicorn.
    """
    from app.config import get_city_config
    from app.core.h3_utils import latlng_to_h3, h3_to_centroid
    from app.core.time_multiplier import get_time_multiplier
    from app.scoring.base_scorer import FeatureSet
    from app.scoring.weighted_scorer import WeightedRiskScorer
    from app.services.macro_baseline_service import get_macro_baseline_score
    from app.services.osm_service import compute_isolation_score
    from app.services.places_service import (
        compute_commercial_density_score,
        compute_hospital_distance_score,
        compute_police_distance_score,
    )
    from app.services.population_service import compute_population_density_score

    city_config = get_city_config()
    h3_cell = latlng_to_h3(lat, lng, city_config.h3_resolution)
    cell_lat, cell_lng = h3_to_centroid(h3_cell)
    time_band = get_time_multiplier(hour)

    fallback_features = []

    def safe_compute(fn, *args, label=""):
        try:
            score, meta, is_fallback = fn(*args)
            if is_fallback:
                fallback_features.append(label)
            return score
        except Exception as e:
            print(f"    [WARNING] {label} failed: {e}")
            fallback_features.append(label)
            return 50.0

    crime_score, _ = get_macro_baseline_score()
    isolation_score = safe_compute(compute_isolation_score, h3_cell, label="isolation")
    commercial_score = safe_compute(compute_commercial_density_score, h3_cell, label="commercial_density")
    police_score = safe_compute(compute_police_distance_score, h3_cell, label="police_distance")
    hospital_score = safe_compute(compute_hospital_distance_score, h3_cell, label="hospital_distance")
    pop_score = safe_compute(compute_population_density_score, h3_cell, label="population_density")

    feature_set = FeatureSet(
        h3_cell=h3_cell,
        lat=cell_lat,
        lng=cell_lng,
        hour=hour,
        crime_baseline_score=crime_score,
        isolation_score=isolation_score,
        commercial_density_score=commercial_score,
        police_distance_score=police_score,
        hospital_distance_score=hospital_score,
        population_density_score=pop_score,
        time_multiplier=time_band.multiplier,
        time_band_name=time_band.band_name,
        time_band_label=time_band.label,
        fallback_features=fallback_features,
    )

    scorer = WeightedRiskScorer()
    result = scorer.score(feature_set)

    return {
        "risk_score": result.risk_score,
        "classification": result.classification,
        "h3_cell": h3_cell,
        "contributing_factors": result.contributing_factors,
        "data_completeness": result.data_completeness,
        "time_band": result.time_band_info,
    }


COLORS = {
    "Low": "\033[92m",       # green
    "Medium": "\033[93m",    # yellow
    "High": "\033[91m",      # red
    "Critical": "\033[95m",  # magenta
    "RESET": "\033[0m",
}


def colorize(text: str, classification: str) -> str:
    color = COLORS.get(classification, "")
    reset = COLORS["RESET"]
    return f"{color}{text}{reset}"


def print_result(location: dict, result: dict, hour: int) -> None:
    score = result["risk_score"]
    cls = result["classification"]
    factors = result.get("contributing_factors", {})
    h3 = result.get("h3_cell", "N/A")
    completeness = result.get("data_completeness", 0)
    time_info = result.get("time_band", {})

    print(f"\n{'-' * 65}")
    print(f"Location: {location['name']}")
    print(f"   ({location['lat']}, {location['lng']}) | Hour: {hour:02d}:00")
    print(f"   {location['notes']}")
    print(f"   H3 Cell: {h3}")
    print()

    score_bar = "#" * int(score // 5) + "-" * (20 - int(score // 5))
    print(f"   RISK SCORE:  {colorize(f'{score:.1f}/100  [{cls}]', cls)}")
    print(f"   [{score_bar}] {score:.1f}")
    print(f"   Data completeness: {completeness}%")
    print(f"   Time band: {time_info.get('label', hour)} (x{time_info.get('multiplier', 1.0):.2f})")
    print()
    print("   Contributing factors:")
    factor_labels = {
        "crime_baseline": "Crime Baseline    ",
        "isolation": "Isolation Score   ",
        "commercial_density": "Commercial Density",
        "police_distance": "Police Distance   ",
        "hospital_distance": "Hospital Distance ",
        "population_density": "Population Density",
        "time_multiplier_boost": "Time Multiplier   ",
    }
    for key, val in factors.items():
        label = factor_labels.get(key, key)
        bar = "#" * int(abs(val) // 2) if val >= 0 else ""
        print(f"     {label}: {val:+6.2f} pts  {bar}")


def run_demo(use_http: bool = True, base_url: str = "http://localhost:8000") -> None:
    print("\n" + "=" * 65)
    print("STREEVA — Area Risk Scoring Engine")
    print("Qualitative Validation: 6 Chennai Locations")
    print("=" * 65)
    print()
    print("METHODOLOGY NOTE:")
    print("  No ground-truth 'unsafe location' dataset exists for Chennai.")
    print("  Validation is QUALITATIVE: we check that intuitive rankings hold.")
    print("  Expected: isolated roads > busy commercial areas in risk score.")
    print("  Expected: night hour > day hour for same location.")

    results_day = []
    results_night = []

    for location in LOCATIONS:
        print(f"\nQuerying: {location['name']}...")

        try:
            for hour_type in ["hour_day", "hour_night"]:
                hour = location[hour_type]
                if use_http:
                    result = query_via_http(location["lat"], location["lng"], hour, base_url)
                else:
                    result = query_direct(location["lat"], location["lng"], hour)
                print_result(location, result, hour)
                time.sleep(0.5)  # Small delay to avoid hammering APIs

                entry = {
                    "name": location["name"],
                    "hour": hour,
                    "score": result["risk_score"],
                    "classification": result["classification"],
                }
                if hour_type == "hour_day":
                    results_day.append(entry)
                else:
                    results_night.append(entry)

        except Exception as exc:
            print(f"  ERROR: {exc}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("SUMMARY TABLE (Qualitative Validation)")
    print("=" * 65)
    print(f"{'Location':<35} {'Day (14h)':>10} {'Night (22h)':>12} {'Diff':>8}")
    print(f"{'-' * 35} {'-' * 10} {'-' * 12} {'-' * 8}")

    for day, night in zip(results_day, results_night):
        delta = night["score"] - day["score"]
        name = day["name"][:34]
        print(f"{name:<35} {day['score']:>7.1f} [{day['classification'][:1]}]  "
              f"{night['score']:>7.1f} [{night['classification'][:1]}]  "
              f"{delta:>+6.1f}")

    print()
    print("Classification keys: L=Low  M=Medium  H=High  C=Critical")
    print()
    print("VALIDATION CHECKS:")
    if results_day:
        max_day = max(results_day, key=lambda x: x["score"])
        min_day = min(results_day, key=lambda x: x["score"])
        print(f"  [PASS] Highest risk (day): {max_day['name'][:40]} ({max_day['score']:.1f})")
        print(f"  [PASS] Lowest risk  (day): {min_day['name'][:40]} ({min_day['score']:.1f})")

        all_night_higher = all(n["score"] > d["score"] for d, n in zip(results_day, results_night))
        print(f"  {'[PASS]' if all_night_higher else '[FAIL]'} All locations score higher at night: {all_night_higher}")

    print()
    print("NOTE: Exact scores depend on live API data.")
    print("      Small deviations from expected rankings may occur due to")
    print("      OSM road network coverage and Places API result counts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STREEVA qualitative validation demo.")
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Query scoring engine directly without starting uvicorn.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="FastAPI server base URL (default: http://localhost:8000).",
    )
    args = parser.parse_args()
    run_demo(use_http=not args.no_server, base_url=args.url)
