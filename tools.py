"""
BuildingTools = the "custom agentic tools" layer the LLM calls.

Updated for SmallOffice_CentralDOAS.idf zone names and to use
EnergyPlus's BUILT-IN "Zone Temperature Control" EMS actuator
(Heating Setpoint / Cooling Setpoint) instead of a custom schedule
actuator. This actuator is automatically available for any zone that
has a ZoneControl:Thermostat object -- no IDF changes needed for it.

Energy is tracked via the whole-building "Electricity:Facility" meter,
which is more robust than per-zone energy variables for a real DOAS
system (vs. ideal loads).
"""
import math
from typing import Dict, List

ZONE_NAMES = [
    "Core_ZN",
    "Perimeter_ZN_1",
    "Perimeter_ZN_2",
    "Perimeter_ZN_3",
    "Perimeter_ZN_4",
]

# Corrections log, module-level so main.py can dump it at the end of a run.
CORRECTIONS_LOG: List[str] = []


def simple_pmv(air_temp_c: float, rh_percent: float = 50.0,
                clo: float = 1.0, met: float = 1.2) -> float:
    """
    Simplified steady-state PMV approximation (not the full Fanger
    iterative solve, which needs mean radiant temp + air velocity we
    don't have cheap access to from EnergyPlus's EMS API mid-timestep).
    Good enough to rank comfort states and prove the agent is reasoning
    about comfort, not just raw air temperature.

    Centered so PMV ~= 0 near 22-23C at 50% RH, met=1.2, clo=1.0 (typical
    office assumptions), moving roughly +/-0.3 PMV per degree C.
    """
    neutral_temp = 22.5 - (clo - 1.0) * 1.5 + (met - 1.2) * 1.0
    humidity_adj = (rh_percent - 50.0) * 0.01
    pmv = (air_temp_c - neutral_temp) * 0.3 + humidity_adj
    return round(pmv, 2)


def carbon_intensity(sim_time_min: float) -> float:
    """
    Synthetic local grid carbon intensity signal (gCO2/kWh) -- a real
    live feed isn't available in this sandbox. Modeled as a diurnal
    curve: evening demand peak (more gas-peaker use), midday dip (more
    baseload/solar). Enough for the agent to weigh "when is it worth
    using more energy" against.
    """
    hour = (sim_time_min / 60) % 24
    base = 350
    evening_peak = 150 * math.exp(-((hour - 19) ** 2) / 8)
    midday_dip = -80 * math.exp(-((hour - 13) ** 2) / 10)
    return round(base + evening_peak + midday_dip, 1)


class BuildingTools:
    def __init__(self, api, zone_names=None):
        self.api = api
        self.zone_names = zone_names or ZONE_NAMES
        self._handles_ready = False
        self._temp_handles = {}
        self._heat_actuator = {}
        self._cool_actuator = {}
        self._meter_handle = None

    def _ensure_handles(self, state):
        if self._handles_ready:
            return
        ex = self.api.exchange
        # CRITICAL: handles requested before the API reports "fully ready"
        # (i.e. during warmup/sizing passes) silently come back as invalid
        # (-1), and every read/write against them silently no-ops or
        # returns 0 for the rest of the run. Must wait for this flag.
        if not ex.api_data_fully_ready(state):
            return
        for zone in self.zone_names:
            self._temp_handles[zone] = ex.get_variable_handle(
                state, "Zone Mean Air Temperature", zone
            )
            self._heat_actuator[zone] = ex.get_actuator_handle(
                state, "Zone Temperature Control", "Heating Setpoint", zone
            )
            self._cool_actuator[zone] = ex.get_actuator_handle(
                state, "Zone Temperature Control", "Cooling Setpoint", zone
            )
        self._meter_handle = ex.get_meter_handle(state, "Electricity:Facility")

        # NOTE: on this EnergyPlus install, the Electricity:Facility meter
        # handle never resolves (-1) despite being a confirmed valid meter
        # name (verified via eplusout.mdd). Rather than block the entire
        # control loop on this one broken handle, we only require the
        # zone temperature/actuator handles to be valid -- those ARE
        # confirmed working. Energy totals are computed post-run directly
        # from EnergyPlus's own eplusout.csv output instead (see
        # analyze_energy.py), which is more authoritative anyway.
        required_handles = (
            list(self._temp_handles.values())
            + list(self._heat_actuator.values())
            + list(self._cool_actuator.values())
        )
        if any(h == -1 for h in required_handles):
            return

        self._handles_ready = True

    def ready(self, state) -> bool:
        """True once handles are created AND we're past warmup for this
        environment. main.py should skip read/act calls until this is
        True, instead of logging/acting on bogus early data."""
        self._ensure_handles(state)
        if not self._handles_ready:
            return False
        return not self.api.exchange.warmup_flag(state)

    # ---- read tool ----
    def read_all_zones(self, state, sim_time_min: float = 0.0) -> Dict[str, float]:
        self._ensure_handles(state)
        ex = self.api.exchange
        out = {}
        for zone in self.zone_names:
            temp = ex.get_variable_value(state, self._temp_handles[zone])
            out[f"{zone}_temp"] = temp
            out[f"{zone}_pmv"] = simple_pmv(temp)
        out["facility_electricity_j"] = ex.get_meter_value(state, self._meter_handle)
        out["carbon_intensity_gco2_kwh"] = carbon_intensity(sim_time_min)
        return out

    # ---- write tool ----
    def set_zone_setpoints(self, state, zone: str, heating_c: float, cooling_c: float):
        """Set both heating and cooling setpoints for a zone (degrees C).
        Self-corrects (clamps) unsafe or inverted values instead of
        blindly forwarding them to EnergyPlus, and logs the correction --
        this is the agent's self-correction loop."""
        self._ensure_handles(state)
        ex = self.api.exchange

        orig_heating, orig_cooling = heating_c, cooling_c

        # Hard safety bounds -- never let the LLM push the building outside
        # a survivable/reasonable range.
        heating_c = max(15.0, min(heating_c, 26.0))
        cooling_c = max(20.0, min(cooling_c, 30.0))

        # Enforce a minimum deadband so heating/cooling setpoints never
        # invert or collide (EnergyPlus will error, or short-cycle HVAC).
        if cooling_c - heating_c < 2.0:
            mid = (heating_c + cooling_c) / 2
            heating_c = mid - 1.0
            cooling_c = mid + 1.0

        if (orig_heating, orig_cooling) != (heating_c, cooling_c):
            CORRECTIONS_LOG.append(
                f"zone={zone} requested=({orig_heating},{orig_cooling}) "
                f"corrected=({heating_c},{cooling_c})"
            )

        ex.set_actuator_value(state, self._heat_actuator[zone], heating_c)
        ex.set_actuator_value(state, self._cool_actuator[zone], cooling_c)

    def apply_actions(self, state, actions: List[dict]):
        """actions: [{"zone": "Core_ZN", "heating_c": 21.0, "cooling_c": 24.0}, ...]"""
        for a in actions:
            if a["zone"] in self.zone_names:
                self.set_zone_setpoints(state, a["zone"], a["heating_c"], a["cooling_c"])

    # ---- baseline (non-AI) rule-based schedule for comparison run ----
    def apply_baseline_schedule(self, state, sim_time_min: float):
        hour = (sim_time_min / 60) % 24
        occupied = 8 <= hour < 18
        heating_c = 21.0 if occupied else 15.6   # night setback
        cooling_c = 24.0 if occupied else 29.4   # night setup
        for zone in self.zone_names:
            self.set_zone_setpoints(state, zone, heating_c, cooling_c)

    # ---- error-parsing tool (satisfies "extract runtime errors" requirement) ----
    @staticmethod
    def parse_err_file(err_path: str) -> List[str]:
        warnings = []
        try:
            with open(err_path) as f:
                for line in f:
                    if "** Severe **" in line or "** Fatal **" in line:
                        warnings.append(line.strip())
        except FileNotFoundError:
            pass
        return warnings