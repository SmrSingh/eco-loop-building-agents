"""
Eco-Loop Building Agents -- main driver.

Runs an EnergyPlus simulation and, on every zone timestep, calls back into
Python so the agent can read sensor data and write actuator (setpoint)
values -- the closed loop:

    EnergyPlus --sensors--> agent --tool calls--> EnergyPlus (actuators)

Requires: EnergyPlus 26.1.0 installed locally.

Usage:
    python main.py --idf SmallOffice_AI.idf --epw weather.epw --mode ai
    python main.py --idf SmallOffice_CentralDOAS.idf --epw weather.epw --mode baseline
"""
import sys
sys.path.insert(0, r"C:\EnergyPlusV26-1-0")

import argparse
import csv
from pathlib import Path

from pyenergyplus.api import EnergyPlusAPI

from tools import BuildingTools, ZONE_NAMES, CORRECTIONS_LOG
from llm_agent import EcoLoopAgent

LOG_PATH = Path("run_log.csv")
DECISION_INTERVAL_SEC = 120 * 60  # ask the agent every 60 sim-minutes, not every timestep


def build_callback(api, mode, tools, agent, log_writer, clock):
    def on_timestep(state):
        if not tools.ready(state):
            return  # still in warmup/sizing -- handles not valid yet, skip

        sim_time_min = api.exchange.current_time(state) * 60
        if sim_time_min - clock["last_decision"] < DECISION_INTERVAL_SEC / 60:
            snapshot = tools.read_all_zones(state, sim_time_min)
            log_writer.writerow({"t_min": sim_time_min, "mode": mode, **snapshot})
            return

        clock["last_decision"] = sim_time_min
        snapshot = tools.read_all_zones(state, sim_time_min)

        if mode == "baseline":
            tools.apply_baseline_schedule(state, sim_time_min)
        else:
            actions = agent.decide(snapshot, sim_time_min)
            tools.apply_actions(state, actions)

        log_writer.writerow({"t_min": sim_time_min, "mode": mode, **snapshot})

    return on_timestep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idf", required=True)
    parser.add_argument("--epw", required=True)
    parser.add_argument("--mode", choices=["baseline", "ai"], default="ai")
    parser.add_argument("--out", default="sim_out")
    args = parser.parse_args()

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    tools = BuildingTools(api, ZONE_NAMES)
    agent = EcoLoopAgent(tools) if args.mode == "ai" else None
    clock = {"last_decision": -9999}

    fieldnames = ["t_min", "mode"] + [f"{z}_temp" for z in ZONE_NAMES] + \
                 [f"{z}_pmv" for z in ZONE_NAMES] + \
                 ["facility_electricity_j", "carbon_intensity_gco2_kwh"]

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        cb = build_callback(api, args.mode, tools, agent, writer, clock)
        api.runtime.callback_end_zone_timestep_after_zone_reporting(state, cb)

        api.runtime.run_energyplus(state, [
            "-w", args.epw,
            "-d", args.out,
            "-r", args.idf,
        ])

    print(f"Done. Log written to {LOG_PATH}")
    if CORRECTIONS_LOG:
        print(f"Agent self-corrections applied: {len(CORRECTIONS_LOG)}")
        with open("corrections.log", "w") as f:
            f.write("\n".join(CORRECTIONS_LOG))
        print("Correction details written to corrections.log")


if __name__ == "__main__":
    main()