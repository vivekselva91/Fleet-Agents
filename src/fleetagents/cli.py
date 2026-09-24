"""Command line entry point.

    python -m fleetagents.cli simulate --vehicles 8 --minutes 90
    python -m fleetagents.cli simulate --explain --trace traces/run.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .domain import TripState, Vehicle
from .fleet import build_scenario, kpis
from .llm import Narrator
from .orchestrator import Orchestrator, Policy
from .telemetry import Trace


def _fleet_table(vehicles: List[Vehicle]) -> str:
    rows = [
        f"{'vehicle':<8} {'state':<18} {'soc':>6} {'trip':<8} {'elapsed':>8} {'promise':>8}"
    ]
    rows.append("-" * 60)
    for v in vehicles:
        t = v.trip
        rows.append(
            f"{v.id:<8} {v.state.value:<18} {v.soc:>5.0%} "
            f"{(t.id if t else '-'):<8} "
            f"{(f'{t.elapsed_minutes:.0f}m' if t else '-'):>8} "
            f"{(f'{t.promised_minutes:.0f}m' if t else '-'):>8}"
        )
    return "\n".join(rows)


def simulate(args: argparse.Namespace) -> int:
    trace = Trace(path=args.trace)
    scenario = build_scenario(
        n_vehicles=args.vehicles,
        seed=args.seed,
        start_hour=args.start_hour,
        low_battery_share=args.low_battery,
    )
    policy = Policy(min_confidence=args.min_confidence)
    orch = Orchestrator(scenario.world, policy=policy, trace=trace)
    narrator = Narrator()

    print(f"scenario : {scenario.name}")
    print(f"           {scenario.description}")
    print(
        f"policy   : confidence floor {policy.min_confidence}, "
        f"reserve SoC {policy.reserve_soc:.0%}, "
        f"max {policy.max_reroutes_per_trip} reroutes/trip\n"
    )

    ticks = int(args.minutes / args.tick)
    for step in range(ticks):
        active = [v for v in scenario.vehicles
                  if v.trip and v.trip.state is not TripState.COMPLETED]
        if not active:
            break
        decisions = orch.step(active, minutes=args.tick)
        if args.explain:
            for d in decisions:
                if d.chosen is None and not d.vetoed:
                    continue
                rec = {
                    "vehicle": d.vehicle_id,
                    "chosen": d.chosen.to_dict() if d.chosen else None,
                    "vetoed": d.vetoed,
                }
                clock = scenario.world.minutes % 1440
                stamp = f"{int(clock // 60):02d}:{int(clock % 60):02d}"
                print(f"  [{stamp}] {narrator.narrate(rec)}")

    print("\n" + _fleet_table(scenario.vehicles))
    results = kpis(scenario.vehicles, orch.stats)
    print("\nKPIs")
    for k, v in results.items():
        print(f"  {k:<18} {v}")

    if args.trace:
        print(f"\ntrace written to {args.trace} ({len(trace.events)} events, "
              f"{len(orch.tools.calls)} tool calls)")
    if args.json:
        print(json.dumps(results, indent=2))
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fleetagents", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run a fleet scenario")
    sim.add_argument("--vehicles", type=int, default=8)
    sim.add_argument("--minutes", type=float, default=90.0, help="simulated duration")
    sim.add_argument("--tick", type=float, default=2.0, help="minutes per decision cycle")
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument("--start-hour", type=float, default=16.0)
    sim.add_argument("--low-battery", type=float, default=0.35,
                     help="share of the fleet starting below 30%% SoC")
    sim.add_argument("--min-confidence", type=float, default=0.45)
    sim.add_argument("--explain", action="store_true", help="narrate every decision")
    sim.add_argument("--trace", type=str, default=None, help="write a JSONL trace here")
    sim.add_argument("--json", action="store_true", help="print KPIs as JSON")
    sim.set_defaults(func=simulate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
