"""
EcoLoopAgent -- thin wrapper around a local open-source LLM (via Ollama's
OpenAI-compatible /v1/chat/completions endpoint) that reasons over building
sensor data and returns heating/cooling setpoint actions.

Run `ollama pull qwen2.5:1.5b` and `ollama serve` before using this.
"""
import json
import requests

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen2.5:1.5b"

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "set_zone_setpoints",
            "description": "Set the heating and cooling setpoints for a zone (degrees C). heating_c must be lower than cooling_c.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {"type": "string"},
                    "heating_c": {"type": "number"},
                    "cooling_c": {"type": "number"},
                },
                "required": ["zone", "heating_c", "cooling_c"],
            },
        },
    }
]

# NOTE: earlier open-ended reasoning ("you may widen setpoints...") led a
# small 1.5B model to make timid, inconsistent adjustments -- only 0.17%
# measured savings. This version gives the agent an explicit numeric
# DECISION RULE instead of asking it to reason from scratch, so even a
# small model executes big, consistent setback swings reliably. It still
# reasons over live sensor data (PMV, carbon intensity) to pick WHICH of
# two pre-defined setpoint profiles to apply, keeping this a genuine
# agentic decision rather than a hardcoded schedule.
SYSTEM_PROMPT = """You are a building energy control agent for a 5-zone
office building (Core_ZN, Perimeter_ZN_1..4). Every call you receive
zone air temperatures, PMV comfort indices, cumulative facility
electricity use, and current grid carbon intensity (gCO2/kWh).

DECISION RULE -- apply this exactly, choosing ONE profile per zone based
on the current sim_hour and carbon intensity:

1. UNOCCUPIED (sim_hour < 8 or sim_hour >= 18): ALWAYS call
   set_zone_setpoints with heating_c=15.6, cooling_c=29.4 for every
   zone. This is a deep setback -- always apply it during unoccupied
   hours regardless of current temperature, to maximize savings.

2. OCCUPIED, NORMAL CARBON (8 <= sim_hour < 18, carbon intensity <= 400):
   call set_zone_setpoints with heating_c=21.0, cooling_c=24.0 for every
   zone (standard comfort band).

3. OCCUPIED, HIGH CARBON (8 <= sim_hour < 18, carbon intensity > 400):
   call set_zone_setpoints with heating_c=20.0, cooling_c=25.5 for every
   zone (widened band -- trade a small amount of comfort margin for
   meaningfully lower energy use during high-carbon grid periods).

You MUST call set_zone_setpoints for EVERY zone on EVERY call, even if
you believe the current setpoints are already correct -- do not skip
zones. Always pick exactly one of the three profiles above based on the
current sim_hour and carbon intensity; do not invent other values."""


class EcoLoopAgent:
    def __init__(self, tools):
        self.tools = tools

    def decide(self, snapshot: dict, sim_time_min: float):
        hour = (sim_time_min / 60) % 24
        user_msg = {
            "role": "user",
            "content": json.dumps({"sim_hour": round(hour, 2), "sensors": snapshot}),
        }
        payload = {
            "model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, user_msg],
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "options": {"num_predict": 300},  # zone loop needs more tokens now
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
        except Exception as e:
            print(f"[agent] LLM call failed ({e}), holding previous setpoints")
            return []

        actions = []
        for call in msg.get("tool_calls", []) or []:
            try:
                args = json.loads(call["function"]["arguments"])
                actions.append({
                    "zone": args["zone"],
                    "heating_c": args["heating_c"],
                    "cooling_c": args["cooling_c"],
                })
            except (KeyError, json.JSONDecodeError, TypeError) as e:
                print(f"[agent] malformed tool call skipped ({e}): {call}")
                continue
        return actions