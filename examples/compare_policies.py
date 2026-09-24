"""Does the agent stack actually earn its keep?

Runs the same seeded scenarios three ways and prints the difference:

  baseline    no agents at all - vehicles drive the direct route, come what may
  route-only  traffic-aware rerouting, nothing else
  full-stack  route + energy + intent + assist under the arbitration policy

Run with:  PYTHONPATH=src python examples/compare_policies.py
"""

from __future__ import annotations

import statistics
from typing import Dict, List

from fleetagents import Orchestrator, Policy, ToolRegistry, build_scenario, kpis
from fleetagents.agents import AssistAgent, EnergyAgent, IntentAgent, RouteAgent
from fleetagents.domain import TripState

SEEDS = [3, 7, 11, 21, 42, 55]
MINUTES = 90.0
TICK = 2.0
VEHICLES = 10


def run(config: str, seed: int) -> Dict[str, float]:
    scenario = build_scenario(n_vehicles=VEHICLES, seed=seed, low_battery_share=0.4)
    tools = ToolRegistry(scenario.world)
    if config == "baseline":
        agents: List = []
    elif config == "route-only":
        agents = [RouteAgent(tools, scenario.world)]
    else:
        agents = [
            RouteAgent(tools, scenario.world),
            EnergyAgent(tools, scenario.world),
            IntentAgent(tools, scenario.world),
            AssistAgent(tools, scenario.world),
        ]
    orch = Orchestrator(scenario.world, tools=tools, agents=agents, policy=Policy())

    for _ in range(int(MINUTES / TICK)):
        active = [v for v in scenario.vehicles
                  if v.trip and v.trip.state is not TripState.COMPLETED]
        if not active:
            break
        orch.step(active, minutes=TICK)

    result = kpis(scenario.vehicles, orch.stats)
    # a vehicle that finishes under the reserve floor is a fleet incident
    result["stranded"] = sum(1 for v in scenario.vehicles if v.soc <= Policy().reserve_soc)
    return result


def main() -> None:
    configs = ["baseline", "route-only", "full-stack"]
    table: Dict[str, Dict[str, float]] = {}

    for config in configs:
        runs = [run(config, s) for s in SEEDS]
        table[config] = {
            "on_time_pct": round(statistics.mean(r["on_time_pct"] for r in runs), 1),
            "avg_minutes": round(statistics.mean(r["avg_minutes"] for r in runs), 1),
            "avg_end_soc": round(statistics.mean(r["avg_end_soc"] for r in runs), 3),
            "stranded": round(statistics.mean(r["stranded"] for r in runs), 2),
            "charge_stops": round(statistics.mean(r["charge_stops"] for r in runs), 1),
            "handoffs": round(statistics.mean(r["handoffs"] for r in runs), 1),
            "reroutes": round(statistics.mean(r["reroutes"] for r in runs), 1),
        }

    cols = ["on_time_pct", "avg_minutes", "avg_end_soc", "stranded",
            "reroutes", "charge_stops", "handoffs"]
    print(f"{VEHICLES} vehicles, {MINUTES:.0f} min, mean over {len(SEEDS)} seeds\n")
    print(f"{'config':<12}" + "".join(f"{c:>14}" for c in cols))
    print("-" * (12 + 14 * len(cols)))
    for config in configs:
        row = table[config]
        print(f"{config:<12}" + "".join(f"{row[c]:>14}" for c in cols))
    print(
        "\nstranded = vehicles finishing at or below the reserve floor. "
        "That column is the one the energy agent exists to hold at zero."
    )


if __name__ == "__main__":
    main()
