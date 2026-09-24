"""Domain objects: geography, vehicles, trips, passenger intent."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# --------------------------------------------------------------------------
# geography
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """A location on the service-area grid, in kilometres from the SW corner."""

    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def toward(self, other: "Point", km: float) -> "Point":
        d = self.distance_to(other)
        if d <= km or d == 0:
            return other
        ratio = km / d
        return Point(self.x + (other.x - self.x) * ratio, self.y + (other.y - self.y) * ratio)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"({self.x:.1f}, {self.y:.1f})"


@dataclass
class Charger:
    id: str
    location: Point
    stalls: int
    kw: float
    reserved: int = 0

    @property
    def available(self) -> int:
        return max(0, self.stalls - self.reserved)


# --------------------------------------------------------------------------
# passengers and trips
# --------------------------------------------------------------------------


class Intent(str, Enum):
    """What the rider actually cares about. Drives arbitration weights."""

    FASTEST = "fastest"          # commuter, hates detours
    COMFORT = "comfort"          # avoids rough weather and hard reroutes
    CHEAPEST = "cheapest"        # tolerant of detours and charge stops
    ACCESSIBILITY = "accessibility"  # needs predictable pickup, no surprises

    @property
    def max_detour_min(self) -> float:
        return {
            Intent.FASTEST: 4.0,
            Intent.COMFORT: 9.0,
            Intent.CHEAPEST: 15.0,
            Intent.ACCESSIBILITY: 6.0,
        }[self]

    @property
    def weather_sensitivity(self) -> float:
        return {
            Intent.FASTEST: 0.3,
            Intent.COMFORT: 1.0,
            Intent.CHEAPEST: 0.4,
            Intent.ACCESSIBILITY: 0.9,
        }[self]


class TripState(str, Enum):
    QUEUED = "queued"
    ENROUTE = "enroute"
    COMPLETED = "completed"
    ESCALATED = "escalated"


@dataclass
class Trip:
    id: str
    origin: Point
    destination: Point
    intent: Intent
    promised_minutes: float
    state: TripState = TripState.QUEUED
    elapsed_minutes: float = 0.0
    detour_minutes: float = 0.0

    @property
    def late_by(self) -> float:
        return max(0.0, self.elapsed_minutes - self.promised_minutes)

    @property
    def on_time(self) -> bool:
        return self.elapsed_minutes <= self.promised_minutes


# --------------------------------------------------------------------------
# vehicles
# --------------------------------------------------------------------------


class VehicleState(str, Enum):
    IDLE = "idle"
    ENROUTE = "enroute"
    CHARGING = "charging"
    AWAITING_OPERATOR = "awaiting_operator"


@dataclass
class Vehicle:
    id: str
    position: Point
    soc: float                      # state of charge, 0.0 - 1.0
    battery_kwh: float = 78.0
    efficiency_kwh_per_km: float = 0.19
    state: VehicleState = VehicleState.IDLE
    trip: Optional[Trip] = None
    waypoints: List[Point] = field(default_factory=list)
    charger_id: Optional[str] = None
    operator_ticket: Optional[str] = None

    @property
    def range_km(self) -> float:
        return (self.soc * self.battery_kwh) / self.efficiency_kwh_per_km

    def energy_for(self, km: float, weather_penalty: float = 0.0) -> float:
        """SoC fraction consumed over `km`, inflated by weather (heating, wipers, caution)."""
        kwh = km * self.efficiency_kwh_per_km * (1.0 + weather_penalty)
        return kwh / self.battery_kwh

    def next_stop(self) -> Optional[Point]:
        return self.waypoints[0] if self.waypoints else None
