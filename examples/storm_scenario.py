"""A single vehicle, a storm, a dying battery and a blocked corridor.

The narrow case that shows arbitration doing its job: the route agent wants a
detour, the energy agent says there is not enough charge for it, the rider
bought the fast option, and the assist agent is watching the clock.

Run with:  PYTHONPATH=src python examples/storm_scenario.py
"""

from __future__ import annotations

from fleetagents import Orchestrator, Point, Policy, Trace, Trip, Vehicle, World
from fleetagents.domain import Intent, TripState, VehicleState
from fleetagents.llm import Narrator
from fleetagents.world import Incident, WeatherCell


def main() -> None:
    world = World(seed=19)
    world.minutes = 17 * 60          # evening peak
    world.weather = [
        WeatherCell(center=Point(10.0, 10.0), radius_km=11.0, severity=0.9,
                    drift_x=2.0, drift_y=0.5)
    ]
    world.incidents = [
        Incident(id="INC-MAIN", location=Point(11.0, 11.0), radius_km=2.6, severity=0.95,
                 expires_at=1e9, description="collision blocking two lanes"),
    ]

    trip = Trip(
        id="T-STORM",
        origin=Point(3.0, 3.0),
        destination=Point(19.0, 19.0),
        intent=Intent.FASTEST,        # 4 minutes of detour tolerance, no more
        promised_minutes=26.0,
        state=TripState.ENROUTE,
    )
    vehicle = Vehicle(id="AV-01", position=trip.origin, soc=0.22,
                      state=VehicleState.ENROUTE, trip=trip)

    trace = Trace()
    orch = Orchestrator(world, policy=Policy(min_confidence=0.45), trace=trace)
    narrator = Narrator()

    print("AV-01: 22% SoC, storm overhead, collision on the direct corridor,")
    print("       rider intent 'fastest' (4 min detour tolerance), 26 min promised.\n")

    for _ in range(45):
        if trip.state is TripState.COMPLETED:
            break
        decisions = orch.step([vehicle], minutes=2.0)
        for d in decisions:
            if d.chosen is None and not d.vetoed:
                continue
            clock = world.minutes % 1440
            rec = {"vehicle": d.vehicle_id,
                   "chosen": d.chosen.to_dict() if d.chosen else None,
                   "vetoed": d.vetoed}
            print(f"[{int(clock // 60):02d}:{int(clock % 60):02d}] {narrator.narrate(rec)}")

    print(f"\nfinished: {trip.state.value}, {trip.elapsed_minutes:.0f} min "
          f"against a {trip.promised_minutes:.0f} min promise, ending at {vehicle.soc:.0%} SoC")
    print(f"decisions traced: {len(trace.of_kind('decision'))}")


if __name__ == "__main__":
    main()
