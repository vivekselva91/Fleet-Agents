"""Optional LLM layer.

The agents reason with explicit policy by default: deterministic, testable, and
free to run in CI. Set FLEET_LLM=anthropic and provide ANTHROPIC_API_KEY to have
an LLM write the operator-facing narrative for a decision instead. The control
flow never depends on it; the model explains, the policy decides.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, Optional


class Narrator:
    """Turns a decision record into a line a remote operator can read."""

    def __init__(self, backend: Optional[str] = None, model: str = "claude-sonnet-4-6") -> None:
        self.backend = backend or os.environ.get("FLEET_LLM", "template")
        self.model = model

    def narrate(self, decision: Dict[str, Any]) -> str:
        if self.backend == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                return self._anthropic(decision)
            except Exception:
                pass  # fall through to the deterministic path
        return self._template(decision)

    # ------------------------------------------------------------------

    @staticmethod
    def _template(d: Dict[str, Any]) -> str:
        chosen = d.get("chosen")
        vetoed_all = d.get("vetoed") or []
        if not chosen:
            if vetoed_all:
                return (
                    f"{d.get('vehicle')}: holding course. Policy vetoed "
                    f"{vetoed_all[0]['action']}: {vetoed_all[0]['reason']}."
                )
            return f"{d.get('vehicle')}: holding course, no agent proposed a change."
        vetoed = vetoed_all
        tail = f" Vetoed: {vetoed[0]['reason']}." if vetoed else ""
        return (
            f"{d.get('vehicle')}: {chosen['action']} on the {chosen['agent']} agent's call "
            f"(utility {chosen['utility']}, confidence {chosen['confidence']}). "
            f"{chosen['rationale']}{tail}"
        )

    def _anthropic(self, d: Dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "max_tokens": 200,
            "system": (
                "You brief remote fleet operators. One or two sentences, plain language, "
                "no speculation beyond the supplied decision record."
            ),
            "messages": [{"role": "user", "content": json.dumps(d, default=str)}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read())
        return "".join(b.get("text", "") for b in body.get("content", [])).strip()
