"""
Computes baseline vs AI energy savings directly from EnergyPlus's own
eplusout.csv output files (NOT the live meter API, which has a broken
handle for Electricity:Facility on this install -- see tools.py comments).

This is actually more authoritative: it reads EnergyPlus's own final
tabulated output rather than a live API value, and sidesteps the broken
meter handle entirely.

Usage (run AFTER both simulations have completed):
    python analyze_energy.py --baseline base_out --ai ai_out
"""
import argparse
import glob
import json
import pandas as pd


def total_electricity_kwh(out_dir: str) -> float:
    csv_path = f"{out_dir}/eplusout.csv"
    df = pd.read_csv(csv_path)

    # Find the Electricity:Facility column -- exact header text can vary
    # slightly by EnergyPlus version (e.g. "Electricity:Facility [J](TimeStep)")
    matches = [c for c in df.columns if c.strip().startswith("Electricity:Facility")]
    if not matches:
        raise RuntimeError(
            f"No 'Electricity:Facility' column found in {csv_path}. "
            f"Columns available: {list(df.columns)[:20]}..."
        )
    col = matches[0]
    total_j = df[col].sum()
    return total_j / 3.6e6  # J -> kWh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="base_out")
    parser.add_argument("--ai", default="ai_out")
    args = parser.parse_args()

    baseline_kwh = total_electricity_kwh(args.baseline)
    ai_kwh = total_electricity_kwh(args.ai)
    pct_reduction = 100 * (baseline_kwh - ai_kwh) / baseline_kwh

    result = {
        "baseline_kwh": round(baseline_kwh, 2),
        "ai_kwh": round(ai_kwh, 2),
        "pct_energy_reduction": round(pct_reduction, 2),
    }
    print(json.dumps(result, indent=2))

    with open("energy_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote energy_summary.json")


if __name__ == "__main__":
    main()
