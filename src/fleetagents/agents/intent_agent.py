"""Passenger intent agent: represents the rider in the arbitration."""

from __future__ import annotations

from typing import Any, Dict, List

from ..domain import Intent, Vehicle
from .base import Action, Agent, Proposal


class IntentAgent(Agent):
    name = "intent"
    charter = ("Speak for the rider: detour tolerance, comfort in bad weather, "
               "and being told what is happening.")

    def propose(self, vehicle: Vehicle, context: Dict[str, Any]) -> List[Proposal]:
        trip = vehicle.trip
        if trip is None:
            return []

        out: List[Proposal] = []
        intent: Intent = trip.intent
        weather = self.call("get_weather", point=vehicle.position)
        severity = weather.data["severity"] if weather.ok else 0.0

        # budget left before the rider's detour tolerance is spent
        remaining = intent.max_detour_min - trip.detour_minutes
        out.append(
            Proposal(
                self.name,
                Action.CONTINUE,
                (
                    f"Rider intent '{intent.value}': {remaining:.1f} min of detour tolerance left, "
                    f"{trip.late_by:.1f} min late so far."
                ),
                utility=0.0,
                confidence=0.9,
                payload={"detour_budget_min": remaining, "veto_over_min": remaining},
            )
        )

        if severity > 0.5 and intent.weather_sensitivity > 0.7:
            out.append(
                Proposal(
                    self.name,
                    Action.NOTIFY,
                    "Heavy weather on route and the rider is weather-sensitive; "
                    "send a proactive update.",
                    utility=2.0,
                    confidence=0.85,
                    payload={"message": "Heavy rain ahead. We're slowing for comfort and safety."},
                )
            )

        if trip.late_by > 5.0:
            out.append(
                Proposal(
                    self.name,
                    Action.NOTIFY,
                    (f"Trip is {trip.late_by:.1f} min past the promise; "
                 "the rider should hear it from us first."),
                    utility=3.0,
                    confidence=0.9,
                    payload={
                    "message": f"Running about {trip.late_by:.0f} min behind. Sorry about that."
                },
                )
            )
        return out
