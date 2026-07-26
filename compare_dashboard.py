"""
Reads two run logs (baseline.csv, ai.csv) plus EnergyPlus's own
eplusout.csv output files (base_out/, ai_out/) and produces:
  1. dashboard.png -- energy comparison + comfort band chart
  2. summary.json  -- headline numbers for your write-up / video

NOTE: energy totals come from eplusout.csv, NOT the facility_electricity_j
column in baseline.csv/ai.csv -- that live meter API handle is broken on
this EnergyPlus install (see tools.py comments), always reads 0. Reading
EnergyPlus's own tabulated output is actually more authoritative anyway.

Usage (run AFTER both simulations have completed):
    python compare_dashboard.py --baseline base_out --ai ai_out
"""
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt


def load(path):
    return pd.read_csv(path)


def temp_cols(df):
    return [c for c in df.columns if c.endswith("_temp")]


def pmv_cols(df):
    return [c for c in df.columns if c.endswith("_pmv")]


def total_electricity_kwh(out_dir: str) -> float:
    df = pd.read_csv(f"{out_dir}/eplusout.csv")
    matches = [c for c in df.columns if c.strip().startswith("Electricity:Facility")]
    if not matches:
        raise RuntimeError(f"No 'Electricity:Facility' column found in {out_dir}/eplusout.csv")
    return df[matches[0]].sum() / 3.6e6  # J -> kWh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="base_out")
    parser.add_argument("--ai", default="ai_out")
    args = parser.parse_args()

    base = load("baseline.csv")
    ai = load("ai.csv")

    base_total_kwh = total_electricity_kwh(args.baseline)
    ai_total_kwh = total_electricity_kwh(args.ai)
    pct_reduction = 100 * (base_total_kwh - ai_total_kwh) / base_total_kwh

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    axes[0].bar(["Baseline", "AI Closed-Loop"], [base_total_kwh, ai_total_kwh],
                color=["#999999", "#2e7d32"])
    axes[0].set_ylabel("Total facility electricity (kWh)")
    axes[0].set_title(f"Energy use: {pct_reduction:.2f}% reduction with AI control")

    for col in temp_cols(ai):
        axes[1].plot(ai["t_min"] / 60, ai[col], label=col, alpha=0.8)
    axes[1].axhspan(21.0, 24.5, color="green", alpha=0.1, label="comfort band")
    axes[1].set_xlabel("Simulation time (hours)")
    axes[1].set_ylabel("Zone temp (C)")
    axes[1].set_title("AI-controlled zone temperatures vs comfort band")
    axes[1].legend(fontsize=7, ncol=3)

    for col in pmv_cols(ai):
        axes[2].plot(ai["t_min"] / 60, ai[col], label=col, alpha=0.8)
    axes[2].axhspan(-0.5, 0.5, color="green", alpha=0.1, label="PMV comfort target")
    axes[2].set_xlabel("Simulation time (hours)")
    axes[2].set_ylabel("PMV")
    axes[2].set_title("AI-controlled Predicted Mean Vote (comfort) vs target band")
    axes[2].legend(fontsize=7, ncol=3)

    plt.tight_layout()
    plt.savefig("dashboard.png", dpi=150)

    try:
        with open("corrections.log") as f:
            corrections_count = len([l for l in f if l.strip()])
    except FileNotFoundError:
        corrections_count = 0

    summary = {
        "baseline_kwh": round(base_total_kwh, 2),
        "ai_kwh": round(ai_total_kwh, 2),
        "pct_energy_reduction": round(pct_reduction, 2),
        "self_corrections_applied": corrections_count,
    }
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Wrote dashboard.png and summary.json")


if __name__ == "__main__":
    main()