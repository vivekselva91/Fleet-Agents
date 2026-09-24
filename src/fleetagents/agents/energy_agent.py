"""Energy agent: owns state of charge, charger reservations and the safety floor."""

from __future__ import annotations

from typing import Any, Dict, List

from ..domain import Point, Vehicle
from .base import Action, Agent, Proposal


class EnergyAgent(Agent):
    name = "energy"
    charter = ("Never let a vehicle arrive below reserve. "
               "Book the cheapest stop that holds that.")

    RESERVE_SOC = 0.15        # hard floor on arrival
    COMFORT_SOC = 0.25        # below this, start looking for a stop
    TARGET_SOC = 0.80

    def propose(self, vehicle: Vehicle, context: Dict[str, Any]) -> List[Proposal]:
        if vehicle.state.value == "charging" or vehicle.charger_id:
            # a stall is already held and the vehicle is on its way to it
            return []

        dest = vehicle.trip.destination if vehicle.trip else None
        if dest is None:
            if vehicle.soc < self.COMFORT_SOC:
                return self._charge_proposals(vehicle, vehicle.position, None, urgent=False)
            return []

        weather = self.call("get_weather", point=vehicle.position)
        penalty = 0.25 * (weather.data["severity"] if weather.ok else 0.0)
        km = vehicle.position.distance_to(dest)
        needed = vehicle.energy_for(km, penalty)
        arrival_soc = vehicle.soc - needed

        if arrival_soc >= self.COMFORT_SOC:
            return [
                Proposal(
                    self.name,
                    Action.CONTINUE,
                    f"Arrives at {arrival_soc:.0%} SoC, above the "
                    f"{self.COMFORT_SOC:.0%} comfort line.",
                    utility=0.0,
                    confidence=0.95,
                )
            ]

        urgent = arrival_soc < self.RESERVE_SOC
        return self._charge_proposals(vehicle, vehicle.position, dest, urgent, arrival_soc)

    # ------------------------------------------------------------------

    def _charge_proposals(
        self,
        vehicle: Vehicle,
        here: Point,
        dest: Point | None,
        urgent: bool,
        arrival_soc: float = 0.0,
    ) -> List[Proposal]:
        found = self.call("find_chargers", point=here, limit=3)
        if not found.ok or not found.data:
            return [
                Proposal(
                    self.name,
                    Action.HANDOFF,
                    "No charger with a free stall in range; needs a human decision.",
                    utility=10.0,
                    confidence=0.9,
                    hard_constraint=urgent,
                    payload={"reason": "no_charger_available"},
                )
            ]

        best = None
        for c in found.data:
            loc: Point = c["location"]
            if dest is not None:
                via = self.call(
                    "estimate_route_via", origin=here, waypoint=loc, destination=dest
                )
                direct = self.leg_estimate(here, dest)
                detour = float(via.data["minutes"]) - direct if via.ok else 99.0
            else:
                detour = self.leg_estimate(here, loc)
            # cheap stalls that are close and fast win
            cost = detour + (30.0 / max(50.0, c["kw"])) * 10.0
            if best is None or cost < best[0]:
                best = (cost, c, detour)

        cost, charger, detour = best
        minutes_charging = self._charge_minutes(vehicle, charger["kw"])
        return [
            Proposal(
                self.name,
                Action.CHARGE_STOP,
                (
                    f"Projected arrival {arrival_soc:.0%} SoC. Stop at {charger['id']} "
                    f"({charger['kw']:.0f} kW): "
                    + (f"+{detour:.1f} min detour" if detour > 0.05 else "on the way, no detour")
                    + f", {minutes_charging:.0f} min plugged in."
                ),
                utility=12.0 if urgent else 5.0,
                confidence=0.92,
                hard_constraint=urgent,
                payload={
                    "charger_id": charger["id"],
                    "waypoint": charger["location"],
                    "detour_min": detour,
                    "charge_min": minutes_charging,
                },
            )
        ]

    def _charge_minutes(self, vehicle: Vehicle, kw: float) -> float:
        delta = max(0.0, self.TARGET_SOC - vehicle.soc)
        kwh = delta * vehicle.battery_kwh
        return (kwh / (kw * 0.8)) * 60.0   # 0.8 derate for taper
