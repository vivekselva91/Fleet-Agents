"""Remote assistance agent: decides when a human should take the decision."""

from __future__ import annotations

from typing import Any, Dict, List

from ..domain import Vehicle
from .base import Action, Agent, Proposal


class AssistAgent(Agent):
    name = "assist"
    charter = "Escalate early and cheaply. A handoff is a cost; a stuck vehicle is a bigger one."

    STUCK_SPEED_KPH = 12.0
    LATE_THRESHOLD_MIN = 12.0

    def propose(self, vehicle: Vehicle, context: Dict[str, Any]) -> List[Proposal]:
        if vehicle.state.value == "awaiting_operator":
            return []

        traffic = self.call("get_traffic", point=vehicle.position, radius_km=1.5)
        weather = self.call("get_weather", point=vehicle.position)
        if not traffic.ok:
            return []

        speed = traffic.data["speed_kph"]
        incidents = traffic.data["incidents"]
        severity = weather.data["severity"] if weather.ok else 0.0
        worst = max((i["severity"] for i in incidents), default=0.0)
        late = vehicle.trip.late_by if vehicle.trip else 0.0

        reasons = []
        if speed < self.STUCK_SPEED_KPH and incidents:
            reasons.append(f"crawling at {speed:.0f} kph beside {incidents[0]['what']}")
        if worst >= 0.85 and severity > 0.45:
            reasons.append("severe incident compounded by storm conditions")
        if late > self.LATE_THRESHOLD_MIN:
            reasons.append(f"{late:.0f} min past the promised arrival")

        if not reasons:
            return []

        sev = "high" if (worst >= 0.85 or speed < 8.0) else "normal"
        return [
            Proposal(
                self.name,
                Action.HANDOFF,
                "Remote operator review: " + "; ".join(reasons) + ".",
                utility=8.0 if sev == "high" else 4.0,
                confidence=0.88,
                hard_constraint=(sev == "high"),
                payload={"reason": reasons[0], "severity": sev},
            )
        ]
