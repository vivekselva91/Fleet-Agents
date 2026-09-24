"""The tool layer.

Agents never touch the world directly. They call tools, every call is recorded,
and each tool declares which agents may use it. In a real deployment these
wrap the traffic API, the weather service, the charging network and the
teleoperations desk; swapping the implementation should not change the agents.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .domain import Point, Vehicle
from .world import World


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: Any
    error: Optional[str] = None


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    allowed_agents: List[str]
    mutating: bool = False


class ToolRegistry:
    """Holds tools, enforces who may call what, and keeps an audit log."""

    def __init__(self, world: World) -> None:
        self.world = world
        self._tools: Dict[str, ToolSpec] = {}
        self.calls: List[Dict[str, Any]] = []
        self._register_defaults()

    # ------------------------------------------------------------ plumbing

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def specs_for(self, agent: str) -> List[ToolSpec]:
        return [s for s in self._tools.values() if agent in s.allowed_agents]

    def call(self, agent: str, name: str, **kwargs: Any) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(name, False, None, f"unknown tool: {name}")
        if agent not in spec.allowed_agents:
            return ToolResult(name, False, None, f"{agent} is not permitted to call {name}")
        try:
            data = spec.fn(**kwargs)
            result = ToolResult(name, True, data)
        except Exception as exc:  # defensive: a bad tool must not kill the run
            result = ToolResult(name, False, None, f"{type(exc).__name__}: {exc}")
        self.calls.append(
            {
                "t": round(self.world.minutes, 1),
                "agent": agent,
                "tool": name,
                "args": {k: str(v) for k, v in kwargs.items()},
                "ok": result.ok,
                "error": result.error,
            }
        )
        return result

    # ------------------------------------------------------------ the tools

    def _register_defaults(self) -> None:
        w = self.world

        def get_traffic(point: Point, radius_km: float = 3.0) -> Dict[str, Any]:
            return {
                "congestion": round(w.congestion_at(point), 3),
                "speed_kph": round(w.speed_kph(point), 1),
                "incidents": [
                    {
                        "id": i.id,
                        "severity": i.severity,
                        "what": i.description,
                        "location": i.location,
                        "radius_km": round(i.radius_km, 2),
                    }
                    for i in w.incidents_near(point, radius_km)
                ],
            }

        def get_weather(point: Point) -> Dict[str, Any]:
            sev = w.weather_at(point)
            label = "clear" if sev < 0.15 else "rain" if sev < 0.5 else "heavy storm"
            return {"severity": round(sev, 3), "conditions": label}

        def estimate_route(origin: Point, destination: Point) -> Dict[str, Any]:
            return {
                "distance_km": round(origin.distance_to(destination), 2),
                "minutes": round(w.travel_minutes(origin, destination), 1),
            }

        def estimate_route_via(
            origin: Point, waypoint: Point, destination: Point
        ) -> Dict[str, Any]:
            legs = w.travel_minutes(origin, waypoint) + w.travel_minutes(waypoint, destination)
            km = origin.distance_to(waypoint) + waypoint.distance_to(destination)
            return {"distance_km": round(km, 2), "minutes": round(legs, 1)}

        def find_chargers(point: Point, limit: int = 3) -> List[Dict[str, Any]]:
            out = []
            for c in w.nearest_chargers(point, limit):
                out.append(
                    {
                        "id": c.id,
                        "km": round(point.distance_to(c.location), 2),
                        "available": c.available,
                        "kw": c.kw,
                        "location": c.location,
                    }
                )
            return out

        def reserve_charger(charger_id: str, vehicle_id: str) -> Dict[str, Any]:
            c = w.charger(charger_id)
            if c is None:
                raise ValueError(f"no such charger: {charger_id}")
            if c.available <= 0:
                raise RuntimeError(f"{charger_id} has no free stall")
            c.reserved += 1
            return {"charger_id": charger_id, "vehicle_id": vehicle_id, "kw": c.kw}

        def release_charger(charger_id: str) -> Dict[str, Any]:
            c = w.charger(charger_id)
            if c is None:
                raise ValueError(f"no such charger: {charger_id}")
            c.reserved = max(0, c.reserved - 1)
            return {"charger_id": charger_id, "available": c.available}

        def notify_passenger(vehicle_id: str, message: str) -> Dict[str, Any]:
            return {"vehicle_id": vehicle_id, "delivered": True, "message": message}

        def page_remote_operator(vehicle_id: str, reason: str, severity: str) -> Dict[str, Any]:
            return {
                "ticket": f"RA-{uuid.uuid4().hex[:6].upper()}",
                "vehicle_id": vehicle_id,
                "reason": reason,
                "severity": severity,
                "queue_minutes": 2.0 if severity == "high" else 5.0,
            }

        specs = [
            ToolSpec("get_traffic", "Live congestion, speed and incidents near a point.",
                     get_traffic, ["route", "dispatch", "assist"]),
            ToolSpec("get_weather", "Precipitation severity and conditions at a point.",
                     get_weather, ["route", "energy", "intent", "assist"]),
            ToolSpec("estimate_route", "Travel time and distance for a direct route.",
                     estimate_route, ["route", "energy", "dispatch"]),
            ToolSpec("estimate_route_via", "Travel time and distance through one waypoint.",
                     estimate_route_via, ["route", "energy"]),
            ToolSpec("find_chargers", "Nearest chargers with free stalls.",
                     find_chargers, ["energy", "dispatch"]),
            ToolSpec("reserve_charger", "Hold a stall for a vehicle.",
                     reserve_charger, ["energy"], mutating=True),
            ToolSpec("release_charger", "Release a held stall.",
                     release_charger, ["energy", "dispatch"], mutating=True),
            ToolSpec("notify_passenger", "Send a message to the rider in a vehicle.",
                     notify_passenger, ["intent", "dispatch"], mutating=True),
            ToolSpec("page_remote_operator", "Open a remote assistance ticket.",
                     page_remote_operator, ["assist"], mutating=True),
        ]
        for s in specs:
            self.register(s)


def vehicle_context(world: World, vehicle: Vehicle) -> Dict[str, Any]:
    """The observation each agent starts its turn from."""
    dest = vehicle.trip.destination if vehicle.trip else None
    return {
        "vehicle_id": vehicle.id,
        "position": vehicle.position,
        "soc": round(vehicle.soc, 3),
        "range_km": round(vehicle.range_km, 1),
        "state": vehicle.state.value,
        "destination": dest,
        "intent": vehicle.trip.intent.value if vehicle.trip else None,
        "elapsed_minutes": round(vehicle.trip.elapsed_minutes, 1) if vehicle.trip else 0.0,
        "promised_minutes": vehicle.trip.promised_minutes if vehicle.trip else 0.0,
        "clock": round(world.minutes, 1),
    }
