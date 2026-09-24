"""Fleet assembly and scenario generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from .domain import Intent, Point, Trip, TripState, Vehicle, VehicleState
from .world import World


@dataclass
class Scenario:
    name: str
    vehicles: List[Vehicle]
    world: World
    description: str = ""


def build_scenario(
    name: str = "rush-hour-storm",
    n_vehicles: int = 8,
    seed: int = 7,
    start_hour: float = 16.0,
    low_battery_share: float = 0.35,
) -> Scenario:
    """A fleet mid-shift: evening peak, a storm front crossing, mixed state of charge."""
    rng = random.Random(seed)
    world = World(seed=seed)
    world.minutes = start_hour * 60.0

    intents = list(Intent)
    vehicles: List[Vehicle] = []
    for i in range(n_vehicles):
        origin = Point(rng.uniform(1, world.size_km - 1), rng.uniform(1, world.size_km - 1))
        dest = Point(rng.uniform(1, world.size_km - 1), rng.uniform(1, world.size_km - 1))
        while origin.distance_to(dest) < 6.0:
            dest = Point(rng.uniform(1, world.size_km - 1), rng.uniform(1, world.size_km - 1))

        low = rng.random() < low_battery_share
        soc = rng.uniform(0.16, 0.30) if low else rng.uniform(0.45, 0.92)
        intent = intents[i % len(intents)]
        promised = world.travel_minutes(origin, dest) * 1.12

        trip = Trip(
            id=f"T-{i+1:03d}",
            origin=origin,
            destination=dest,
            intent=intent,
            promised_minutes=round(promised, 1),
            state=TripState.ENROUTE,
        )
        vehicles.append(
            Vehicle(
                id=f"AV-{i+1:02d}",
                position=origin,
                soc=round(soc, 3),
                state=VehicleState.ENROUTE,
                trip=trip,
            )
        )

    return Scenario(
        name=name,
        vehicles=vehicles,
        world=world,
        description=(
            f"{n_vehicles} vehicles, evening peak from {start_hour:.0f}:00, storm front "
            f"crossing west to east, {int(low_battery_share*100)}% of the fleet below 30% SoC."
        ),
    )


def kpis(vehicles: List[Vehicle], stats: dict) -> dict:
    trips = [v.trip for v in vehicles if v.trip]
    done = [t for t in trips if t.state is TripState.COMPLETED]
    on_time = [t for t in done if t.on_time]
    return {
        "trips": len(trips),
        "completed": len(done),
        "on_time_pct": round(100.0 * len(on_time) / len(done), 1) if done else 0.0,
        "avg_minutes": round(sum(t.elapsed_minutes for t in done) / len(done), 1) if done else 0.0,
        "avg_detour_min": (
            round(sum(t.detour_minutes for t in trips) / len(trips), 2) if trips else 0.0
        ),
        "avg_end_soc": round(sum(v.soc for v in vehicles) / len(vehicles), 3) if vehicles else 0.0,
        "reroutes": stats.get("reroutes", 0),
        "charge_stops": stats.get("charge_stops", 0),
        "handoffs": stats.get("handoffs", 0),
        "notifications": stats.get("notifications", 0),
        "policy_vetoes": stats.get("vetoes", 0),
    }
