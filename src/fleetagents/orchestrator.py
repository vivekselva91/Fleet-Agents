"""The coordination engine.

Each tick, for each vehicle:

  1. every agent observes and returns Proposals
  2. guardrails filter proposals that violate policy
  3. arbitration picks one action, hard constraints first, then best score
  4. the chosen action is executed and the world advances

Nothing is applied that did not survive arbitration, and every step is written
to the trace, so any decision in a run can be reconstructed afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .agents import Action, Agent, AssistAgent, EnergyAgent, IntentAgent, Proposal, RouteAgent
from .domain import Point, TripState, Vehicle, VehicleState
from .telemetry import Trace
from .tools import ToolRegistry, vehicle_context
from .world import World


@dataclass
class Policy:
    """Fleet-level rules the agents are not allowed to argue with."""

    min_confidence: float = 0.45        # below this, a human decides instead
    max_reroutes_per_trip: int = 4
    reserve_soc: float = 0.15
    respect_detour_budget: bool = True
    # a condition that is already being handled should not be re-decided every
    # tick: minutes to wait before the same action may fire again per vehicle
    cooldown_minutes: Dict[str, float] = field(
        default_factory=lambda: {
            "reroute": 6.0,
            "charge_stop": 20.0,
            "notify": 12.0,
            "handoff": 15.0,
        }
    )


@dataclass
class Decision:
    vehicle_id: str
    chosen: Optional[Proposal]
    considered: List[Proposal] = field(default_factory=list)
    vetoed: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""


class Orchestrator:
    def __init__(
        self,
        world: World,
        tools: Optional[ToolRegistry] = None,
        agents: Optional[Sequence[Agent]] = None,
        policy: Optional[Policy] = None,
        trace: Optional[Trace] = None,
    ) -> None:
        self.world = world
        self.tools = tools or ToolRegistry(world)
        self.policy = policy or Policy()
        self.trace = trace or Trace()
        # note: an explicitly empty list means "no agents", not "use defaults"
        self.agents: List[Agent] = list(agents) if agents is not None else [
            RouteAgent(self.tools, world),
            EnergyAgent(self.tools, world),
            IntentAgent(self.tools, world),
            AssistAgent(self.tools, world),
        ]
        self.reroute_counts: Dict[str, int] = {}
        self.last_action_at: Dict[str, float] = {}
        self.stats: Dict[str, int] = {
            "reroutes": 0,
            "charge_stops": 0,
            "handoffs": 0,
            "notifications": 0,
            "vetoes": 0,
        }

    # ------------------------------------------------------------ one tick

    def step(self, vehicles: List[Vehicle], minutes: float = 2.0) -> List[Decision]:
        decisions: List[Decision] = []
        for v in vehicles:
            decisions.append(self._decide_and_act(v, minutes))
        self.world.tick(minutes)
        for v in vehicles:
            self._advance(v, minutes)
        return decisions

    def _decide_and_act(self, vehicle: Vehicle, minutes: float) -> Decision:
        if vehicle.state in (VehicleState.CHARGING, VehicleState.AWAITING_OPERATOR):
            return Decision(vehicle.id, None, note=f"vehicle {vehicle.state.value}")

        context = vehicle_context(self.world, vehicle)
        proposals: List[Proposal] = []
        for agent in self.agents:
            try:
                proposals.extend(agent.propose(vehicle, context))
            except Exception as exc:  # one bad agent must not stop the fleet
                self.trace.event(
                    "agent_error", vehicle=vehicle.id, agent=agent.name, error=str(exc)
                )

        kept, vetoed = self._guardrails(vehicle, proposals)
        chosen = self._arbitrate(kept)
        decision = Decision(vehicle.id, chosen, kept, vetoed)

        if chosen is not None:
            self._execute(vehicle, chosen)
            self.last_action_at[f"{vehicle.id}:{chosen.action.value}"] = self.world.minutes

        self.trace.event(
            "decision",
            t=round(self.world.minutes, 1),
            vehicle=vehicle.id,
            soc=round(vehicle.soc, 3),
            chosen=chosen.to_dict() if chosen else None,
            considered=[p.to_dict() for p in kept],
            vetoed=vetoed,
        )
        return decision

    # ---------------------------------------------------------- guardrails

    def _guardrails(self, vehicle: Vehicle, proposals: List[Proposal]):
        kept: List[Proposal] = []
        vetoed: List[Dict[str, Any]] = []
        budget = self._detour_budget(vehicle, proposals)

        for p in proposals:
            reason = None
            cooldown = self.policy.cooldown_minutes.get(p.action.value)
            last = self.last_action_at.get(f"{vehicle.id}:{p.action.value}")

            if cooldown and last is not None and self.world.minutes - last < cooldown:
                reason = (
                    f"{p.action.value} already taken {self.world.minutes - last:.0f} min ago, "
                    f"cooldown {cooldown:.0f} min"
                )

            elif p.confidence < self.policy.min_confidence and not p.hard_constraint:
                reason = (
                    f"confidence {p.confidence:.2f} below floor "
                    f"{self.policy.min_confidence:.2f}"
                )

            elif p.action is Action.REROUTE:
                used = self.reroute_counts.get(vehicle.id, 0)
                if used >= self.policy.max_reroutes_per_trip:
                    reason = f"reroute limit reached ({used})"
                elif self.policy.respect_detour_budget and budget is not None:
                    cost = float(p.payload.get("gain_min", 0.0))
                    # a reroute that loses time is a detour against the rider's budget
                    if -cost > budget:
                        reason = f"detour {-cost:.1f} min exceeds rider budget {budget:.1f} min"

            elif p.action is Action.CHARGE_STOP and not p.hard_constraint:
                detour = float(p.payload.get("detour_min", 0.0))
                if budget is not None and detour > budget:
                    reason = f"charge detour {detour:.1f} min exceeds rider budget {budget:.1f} min"

            if reason:
                self.stats["vetoes"] += 1
                vetoed.append({"agent": p.agent, "action": p.action.value, "reason": reason})
            else:
                kept.append(p)
        return kept, vetoed

    @staticmethod
    def _detour_budget(vehicle: Vehicle, proposals: List[Proposal]) -> Optional[float]:
        for p in proposals:
            if p.agent == "intent" and "detour_budget_min" in p.payload:
                return float(p.payload["detour_budget_min"])
        return None

    # --------------------------------------------------------- arbitration

    def _arbitrate(self, proposals: List[Proposal]) -> Optional[Proposal]:
        if not proposals:
            return None
        hard = [p for p in proposals if p.hard_constraint]
        if hard:
            return max(hard, key=lambda p: p.score())
        actionable = [p for p in proposals if p.action is not Action.CONTINUE and p.score() > 0]
        if not actionable:
            return None
        return max(actionable, key=lambda p: p.score())

    # ----------------------------------------------------------- execution

    def _execute(self, vehicle: Vehicle, p: Proposal) -> None:
        if p.action is Action.REROUTE:
            wp: Point = p.payload["waypoint"]
            vehicle.waypoints = [wp]
            self.reroute_counts[vehicle.id] = self.reroute_counts.get(vehicle.id, 0) + 1
            self.stats["reroutes"] += 1

        elif p.action is Action.CHARGE_STOP:
            res = self.tools.call("energy", "reserve_charger",
                                  charger_id=p.payload["charger_id"], vehicle_id=vehicle.id)
            if res.ok:
                vehicle.waypoints = [p.payload["waypoint"]]
                vehicle.charger_id = p.payload["charger_id"]
                self.stats["charge_stops"] += 1
                if vehicle.trip:
                    vehicle.trip.detour_minutes += float(p.payload.get("detour_min", 0.0))

        elif p.action is Action.NOTIFY:
            self.tools.call("intent", "notify_passenger",
                            vehicle_id=vehicle.id, message=p.payload.get("message", ""))
            self.stats["notifications"] += 1

        elif p.action is Action.HANDOFF:
            res = self.tools.call("assist", "page_remote_operator", vehicle_id=vehicle.id,
                                  reason=p.payload.get("reason", p.rationale),
                                  severity=p.payload.get("severity", "normal"))
            if res.ok:
                vehicle.state = VehicleState.AWAITING_OPERATOR
                vehicle.operator_ticket = res.data["ticket"]
                vehicle.waypoints = []
                self.stats["handoffs"] += 1
                if vehicle.trip:
                    vehicle.trip.state = TripState.ESCALATED

    # ------------------------------------------------------------ movement

    def _advance(self, vehicle: Vehicle, minutes: float) -> None:
        if vehicle.trip and vehicle.trip.state is TripState.ENROUTE:
            vehicle.trip.elapsed_minutes += minutes

        if vehicle.state is VehicleState.AWAITING_OPERATOR:
            # the operator resolves and hands control back
            vehicle.state = VehicleState.ENROUTE
            vehicle.operator_ticket = None
            if vehicle.trip and vehicle.trip.state is TripState.ESCALATED:
                vehicle.trip.state = TripState.ENROUTE
            return

        if vehicle.state is VehicleState.CHARGING:
            charger = self.world.charger(vehicle.charger_id) if vehicle.charger_id else None
            kw = charger.kw * 0.8 if charger else 100.0
            vehicle.soc = min(1.0, vehicle.soc + (kw * (minutes / 60.0)) / vehicle.battery_kwh)
            if vehicle.soc >= EnergyAgent.TARGET_SOC:
                if vehicle.charger_id:
                    self.tools.call("energy", "release_charger", charger_id=vehicle.charger_id)
                vehicle.charger_id = None
                vehicle.state = VehicleState.ENROUTE if vehicle.trip else VehicleState.IDLE
            return

        target = vehicle.next_stop() or (vehicle.trip.destination if vehicle.trip else None)
        if target is None:
            return

        speed = self.world.speed_kph(vehicle.position)
        km = speed * (minutes / 60.0)
        weather_penalty = 0.25 * self.world.weather_at(vehicle.position)
        step_km = min(km, vehicle.position.distance_to(target))
        vehicle.soc = max(0.0, vehicle.soc - vehicle.energy_for(step_km, weather_penalty))
        vehicle.position = vehicle.position.toward(target, km)

        if vehicle.position.distance_to(target) < 0.2:
            if vehicle.waypoints and target is vehicle.waypoints[0]:
                vehicle.waypoints.pop(0)
                if vehicle.charger_id:
                    vehicle.state = VehicleState.CHARGING
                    return
            if vehicle.trip and vehicle.position.distance_to(vehicle.trip.destination) < 0.4:
                vehicle.trip.state = TripState.COMPLETED
                vehicle.state = VehicleState.IDLE
                self.trace.event(
                    "trip_completed",
                    t=round(self.world.minutes, 1),
                    vehicle=vehicle.id,
                    trip=vehicle.trip.id,
                    minutes=round(vehicle.trip.elapsed_minutes, 1),
                    promised=vehicle.trip.promised_minutes,
                    on_time=vehicle.trip.on_time,
                )
