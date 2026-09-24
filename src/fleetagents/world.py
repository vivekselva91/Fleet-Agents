"""A deterministic, seeded simulation of the service area.

The world stands in for the live feeds a real deployment would read: traffic,
weather radar, charger availability, incident reports. Everything is seeded so
a run can be replayed exactly from its trace.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .domain import Charger, Point


@dataclass
class WeatherCell:
    """A storm front drifting across the grid."""

    center: Point
    radius_km: float
    severity: float          # 0.0 clear .. 1.0 severe
    drift_x: float
    drift_y: float

    def advance(self, minutes: float) -> None:
        hours = minutes / 60.0
        self.center = Point(
            self.center.x + self.drift_x * hours,
            self.center.y + self.drift_y * hours,
        )

    def severity_at(self, p: Point) -> float:
        d = self.center.distance_to(p)
        if d >= self.radius_km:
            return 0.0
        falloff = 1.0 - (d / self.radius_km)
        return self.severity * falloff


@dataclass
class Incident:
    """A blocked or degraded corridor: construction, collision, closure."""

    id: str
    location: Point
    radius_km: float
    severity: float
    expires_at: float
    description: str


@dataclass
class World:
    size_km: float = 24.0
    seed: int = 7
    minutes: float = 0.0
    chargers: List[Charger] = field(default_factory=list)
    weather: List[WeatherCell] = field(default_factory=list)
    incidents: List[Incident] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)
    _incident_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if not self.chargers:
            self.chargers = self._seed_chargers()
        if not self.weather:
            self.weather = [
                WeatherCell(
                    center=Point(-4.0, 6.0),
                    radius_km=9.0,
                    severity=0.85,
                    drift_x=5.5,
                    drift_y=1.2,
                )
            ]

    # ---------------------------------------------------------------- setup

    def _seed_chargers(self) -> List[Charger]:
        spots = [
            ("CHG-NORTH", Point(6.0, 19.0), 6, 150.0),
            ("CHG-DOWNTOWN", Point(12.0, 12.0), 10, 250.0),
            ("CHG-EAST", Point(20.0, 9.0), 4, 150.0),
            ("CHG-DEPOT", Point(3.0, 4.0), 12, 350.0),
        ]
        return [Charger(id=i, location=p, stalls=s, kw=k) for i, p, s, k in spots]

    # ---------------------------------------------------------------- clock

    def tick(self, minutes: float) -> None:
        self.minutes += minutes
        for cell in self.weather:
            cell.advance(minutes)
        self.incidents = [i for i in self.incidents if i.expires_at > self.minutes]
        # incidents appear at roughly one per 25 simulated minutes
        if self._rng.random() < minutes / 25.0:
            self._spawn_incident()

    def _spawn_incident(self) -> None:
        self._incident_counter += 1
        kinds = [
            ("collision blocking two lanes", 0.9, 35.0),
            ("construction lane closure", 0.5, 120.0),
            ("event crowd, pedestrian density high", 0.6, 60.0),
            ("signal outage at intersection", 0.7, 45.0),
        ]
        desc, sev, life = self._rng.choice(kinds)
        # Two thirds of incidents land on the busy core, where most routes cross.
        # Uniform placement would put nearly every incident somewhere no vehicle
        # is driving, which is not what a dense service area looks like.
        if self._rng.random() < 0.66:
            mid = self.size_km / 2
            loc = Point(
                min(self.size_km - 1, max(1.0, self._rng.gauss(mid, 4.0))),
                min(self.size_km - 1, max(1.0, self._rng.gauss(mid, 4.0))),
            )
        else:
            loc = Point(
                self._rng.uniform(2, self.size_km - 2),
                self._rng.uniform(2, self.size_km - 2),
            )
        self.incidents.append(
            Incident(
                id=f"INC-{self._incident_counter:03d}",
                location=loc,
                radius_km=self._rng.uniform(1.2, 3.0),
                severity=sev,
                expires_at=self.minutes + life,
                description=desc,
            )
        )

    # ------------------------------------------------------------- readings

    def weather_at(self, p: Point) -> float:
        return max((c.severity_at(p) for c in self.weather), default=0.0)

    def incidents_near(self, p: Point, km: float = 3.0) -> List[Incident]:
        return [i for i in self.incidents if i.location.distance_to(p) <= i.radius_km + km]

    def congestion_at(self, p: Point) -> float:
        """0.0 free flow .. 1.0 gridlock. Time of day plus a stable spatial pattern."""
        hour = (self.minutes / 60.0) % 24.0
        peak = math.exp(-((hour - 8.0) ** 2) / 4.0) + math.exp(-((hour - 17.5) ** 2) / 4.0)
        centre = Point(self.size_km / 2, self.size_km / 2)
        density = max(0.0, 1.0 - centre.distance_to(p) / (self.size_km / 1.6))
        local = 0.25 + 0.55 * density * min(1.0, peak)
        for inc in self.incidents:
            d = inc.location.distance_to(p)
            if d <= inc.radius_km:
                local += inc.severity * (1.0 - d / inc.radius_km)
        return min(1.0, local)

    def speed_kph(self, p: Point) -> float:
        """Effective speed at a point, degraded by congestion and weather."""
        base = 52.0
        congestion = self.congestion_at(p)
        weather = self.weather_at(p)
        return max(8.0, base * (1.0 - 0.65 * congestion) * (1.0 - 0.35 * weather))

    def travel_minutes(self, a: Point, b: Point, samples: int = 0) -> float:
        """Estimate travel time by sampling conditions along the straight-line path.

        Sampling roughly every kilometre matters: a 1.5 km jam on a 20 km route
        disappears entirely if you only probe five points, and the route agent
        would never see a reason to steer around it.
        """
        km = a.distance_to(b)
        if km == 0:
            return 0.0
        if samples <= 0:
            samples = max(6, min(40, int(km * 1.5)))
        speeds = []
        for i in range(samples):
            t = (i + 0.5) / samples
            probe = Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
            speeds.append(self.speed_kph(probe))
        harmonic = len(speeds) / sum(1.0 / s for s in speeds)
        return (km / harmonic) * 60.0

    def nearest_chargers(self, p: Point, limit: int = 3) -> List[Charger]:
        ranked = sorted(self.chargers, key=lambda c: c.location.distance_to(p))
        return [c for c in ranked if c.available > 0][:limit]

    def charger(self, charger_id: str) -> Optional[Charger]:
        return next((c for c in self.chargers if c.id == charger_id), None)

    def snapshot(self) -> Dict[str, object]:
        return {
            "minutes": round(self.minutes, 1),
            "incidents": [i.id for i in self.incidents],
            "storm_center": str(self.weather[0].center) if self.weather else None,
            "chargers_free": {c.id: c.available for c in self.chargers},
        }
