"""Agent contract.

Every agent follows the same loop: observe, call the tools it is allowed to
call, and return zero or more Proposals. Agents never mutate vehicle state
themselves. The orchestrator arbitrates and applies. That split is what keeps
the system auditable: proposals are the diff, arbitration is the review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from ..domain import Point, Vehicle
from ..tools import ToolRegistry
from ..world import World


class Action(str, Enum):
    CONTINUE = "continue"
    REROUTE = "reroute"
    CHARGE_STOP = "charge_stop"
    NOTIFY = "notify"
    HANDOFF = "handoff"
    HOLD = "hold"


@dataclass
class Proposal:
    """One agent's recommendation for one vehicle on one tick."""

    agent: str
    action: Action
    rationale: str
    utility: float = 0.0            # expected benefit, arbitrary but comparable units
    confidence: float = 0.8         # 0..1, below policy floor forces human review
    hard_constraint: bool = False   # safety / energy floor: cannot be outvoted
    payload: Dict[str, Any] = field(default_factory=dict)

    def score(self) -> float:
        return self.utility * self.confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "action": self.action.value,
            "rationale": self.rationale,
            "utility": round(self.utility, 2),
            "confidence": round(self.confidence, 2),
            "hard": self.hard_constraint,
            "payload": {k: str(v) for k, v in self.payload.items()},
        }


class Agent:
    """Base class. Subclasses implement `propose`."""

    name: str = "agent"
    charter: str = ""

    def __init__(self, tools: ToolRegistry, world: World) -> None:
        self.tools = tools
        self.world = world

    def call(self, tool: str, **kwargs: Any):
        return self.tools.call(self.name, tool, **kwargs)

    def propose(self, vehicle: Vehicle, context: Dict[str, Any]) -> List[Proposal]:
        raise NotImplementedError

    # helper shared by several agents
    def leg_estimate(self, a: Point, b: Point) -> float:
        res = self.call("estimate_route", origin=a, destination=b)
        return float(res.data["minutes"]) if res.ok else self.world.travel_minutes(a, b)
