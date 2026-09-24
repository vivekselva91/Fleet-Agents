"""Structured, replayable tracing.

Every decision, tool call and completed trip lands here as one JSON object per
line. A run can be diffed against another run, or replayed into a dashboard.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Trace:
    events: List[Dict[str, Any]] = field(default_factory=list)
    path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.path:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)

    def event(self, kind: str, **fields: Any) -> Dict[str, Any]:
        rec = {"kind": kind, **fields}
        self.events.append(rec)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    def of_kind(self, kind: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("kind") == kind]

    def write(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for e in self.events:
                fh.write(json.dumps(e, default=str) + "\n")
        return path

    @staticmethod
    def new_run_path(directory: str = "traces") -> str:
        return os.path.join(directory, f"run-{int(time.time())}.jsonl")
