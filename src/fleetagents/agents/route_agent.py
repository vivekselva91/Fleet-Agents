"""Route agent: watches traffic and incidents, proposes reroutes."""

from __future__ import annotations

from typing import Any, Dict, List

from ..domain import Point, Vehicle
from .base import Action, Agent, Proposal


class RouteAgent(Agent):
    name = "route"
    charter = "Keep the vehicle moving on the fastest safe corridor available right now."

    # a detour is only worth proposing if it beats the direct path by this much
    MIN_GAIN_MIN = 1.5

    def propose(self, vehicle: Vehicle, context: Dict[str, Any]) -> List[Proposal]:
        if vehicle.trip is None or vehicle.state.value == "charging":
            return []

        dest: Point = vehicle.trip.destination
        weather = self.call("get_weather", point=vehicle.position)

        # Scan ahead along the corridor, not just around the vehicle. A jam
        # eight kilometres out is exactly the one worth routing around, and it
        # is invisible if you only sample where you already are.
        incidents, congestion = self._scan_path(vehicle.position, dest)
        if incidents is None:
            return []

        direct = self.leg_estimate(vehicle.position, dest)
        severity = weather.data["severity"] if weather.ok else 0.0

        if not incidents and congestion < 0.55:
            return [
                Proposal(
                    self.name,
                    Action.CONTINUE,
                    f"Corridor clear: congestion {congestion:.2f}, ETA {direct:.1f} min.",
                    utility=0.0,
                    confidence=0.95,
                )
            ]

        # Candidate waypoints: tight passes either side of each incident, plus a
        # couple of generic lateral options. Tight passes matter - a wide detour
        # almost always costs more time than the jam it avoids.
        candidates = []
        for inc in incidents:
            clearance = float(inc["radius_km"]) + 0.7
            for sign in (1.0, -1.0):
                candidates.append(
                    self._offset_from(vehicle.position, dest, inc["location"], sign * clearance)
                )
        for offset in (-2.0, 2.0):
            candidates.append(self._lateral_waypoint(vehicle.position, dest, offset))

        best = None
        for mid in candidates:
            res = self.call(
                "estimate_route_via", origin=vehicle.position, waypoint=mid, destination=dest
            )
            if not res.ok:
                continue
            minutes = float(res.data["minutes"])
            if best is None or minutes < best[0]:
                best = (minutes, mid)

        if best is None:
            return []

        minutes, waypoint = best
        gain = direct - minutes
        worst = max((i["severity"] for i in incidents), default=0.0)

        if gain < self.MIN_GAIN_MIN:
            # nothing better exists: say so, and let the assist agent see the pain
            return [
                Proposal(
                    self.name,
                    Action.CONTINUE,
                    (
                        f"No better corridor: direct {direct:.1f} min is within "
                        f"{max(0.0, -gain):.1f} min of every alternative."
                    ),
                    utility=0.0,
                    confidence=0.55 if worst >= 0.85 else 0.75,
                    payload={"blocked": worst >= 0.85, "eta_min": direct},
                )
            ]

        # confidence falls when the picture is messy: severe incident plus storm
        confidence = 0.9 - 0.25 * worst - 0.2 * severity
        return [
            Proposal(
                self.name,
                Action.REROUTE,
                (
                    f"Detour saves {gain:.1f} min around "
                    f"{incidents[0]['what'] if incidents else 'congestion'}."
                ),
                utility=gain,
                confidence=max(0.2, confidence),
                payload={"waypoint": waypoint, "eta_min": minutes, "gain_min": gain},
            )
        ]

    def _scan_path(self, here: Point, dest: Point):
        """Probe traffic at the vehicle and at points ahead on the route."""
        seen = {}
        worst_congestion = 0.0
        for frac in (0.0, 0.3, 0.6, 0.85):
            probe = Point(here.x + (dest.x - here.x) * frac, here.y + (dest.y - here.y) * frac)
            res = self.call("get_traffic", point=probe, radius_km=2.0)
            if not res.ok:
                continue
            worst_congestion = max(worst_congestion, float(res.data["congestion"]))
            for inc in res.data["incidents"]:
                seen[inc["id"]] = inc
        if not seen and worst_congestion == 0.0:
            return None, 0.0
        return list(seen.values()), worst_congestion

    @staticmethod
    def _offset_from(a: Point, b: Point, obstacle: Point, offset_km: float) -> Point:
        """A waypoint beside the obstacle, perpendicular to the direction of travel."""
        dx, dy = b.x - a.x, b.y - a.y
        norm = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        px, py = -dy / norm, dx / norm
        return Point(obstacle.x + px * offset_km, obstacle.y + py * offset_km)

    @staticmethod
    def _lateral_waypoint(a: Point, b: Point, offset_km: float) -> Point:
        mid = Point((a.x + b.x) / 2, (a.y + b.y) / 2)
        dx, dy = b.x - a.x, b.y - a.y
        norm = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        # perpendicular unit vector
        px, py = -dy / norm, dx / norm
        return Point(mid.x + px * offset_km, mid.y + py * offset_km)
