# Architecture notes

## Why proposals instead of direct action

The obvious design is to let each agent act: the route agent sets waypoints, the
energy agent books chargers, the assist agent pages an operator. It works until
two agents act in the same tick and the vehicle is simultaneously detouring
around a collision and driving to a charger in the opposite direction.

Here agents return `Proposal` objects and never mutate state. One component, the
orchestrator, decides and writes. That buys three things:

1. **Conflicts are resolved in one place**, with a policy you can read.
2. **Every decision has a recorded alternative.** The trace holds what was
   proposed, what was vetoed and why, not just what happened.
3. **Agents are trivially testable.** Feed a vehicle and a world, assert on the
   proposals. No mocking of side effects.

## Arbitration order

```
1. drop proposals that fail a guardrail        (reason recorded)
2. if any hard constraint survives             → highest utility × confidence among those
3. otherwise, among actionable proposals       → highest utility × confidence
4. nothing actionable                          → vehicle continues
```

`utility` is expected benefit in loosely comparable units — minutes saved for a
reroute, risk avoided for a charge stop. `confidence` is the agent's own read on
how well it understands the situation. Multiplying them means a confident small
win beats a wild guess at a big one.

Hard constraints are the escape hatch for physics and safety: arriving below the
reserve state of charge, or a severe blockage with a stopped vehicle. They cannot
be outvoted by any utility score.

## Confidence as the escalation trigger

Agents lower their own confidence when the inputs conflict — a severe incident
plus heavy weather, for example. When the best available proposal sits below
`Policy.min_confidence`, it is vetoed rather than executed, which leaves either
a safer alternative or nothing. Combined with the assist agent's own escalation
rules, that produces the handoff behaviour without a separate "am I confused?"
subsystem.

Tuning `min_confidence` moves the whole fleet along the autonomy/teleop tradeoff
in one number. Raise it and more decisions reach a human.

## Cooldowns, and why they are policy rather than agent logic

A condition that persists across ticks — a rider running late, a jam that has
not cleared — will produce the same proposal every cycle. Without a cooldown the
rider gets a text every two minutes and the teleop desk gets a new ticket for a
vehicle already in the queue.

Cooldowns live in `Policy`, not inside the agents, because they are a fleet
operations decision rather than a reasoning one. The right notification cadence
is a product question and will change without any agent changing.

## The tool seam

Agents reach the world only through `ToolRegistry`. Each tool declares which
agents may call it, and every call is recorded with arguments and result. Two
consequences worth stating:

- The route agent **cannot** reserve a charger or page an operator. Capability is
  enforced, not merely conventional.
- Replacing the simulation with production feeds is a rewrite of `tools.py`.
  No agent changes.

That is the same boundary an LLM-driven version would use: the tool schemas are
already the function-calling contract.

## Where the LLM belongs

The control flow is deterministic policy, not model output, because a robotaxi
fleet should not have its charge-stop decisions depend on sampling temperature.
The model's job in `llm.py` is narration: turning a decision record into a line a
remote operator can read at a glance.

A reasonable next step is model-assisted proposals for the genuinely ambiguous
cases — unusual incidents, unstructured rider requests — with the same
guardrails and arbitration applied to whatever it returns. The architecture
already supports it: the model would simply be another agent returning
proposals, subject to the same confidence floor.

## Known simplifications

- **Straight-line geography.** There is no road graph; travel time comes from
  sampling conditions along a line. Real deployment needs an actual routing
  engine, which slots in behind `estimate_route`.
- **One vehicle, one trip.** No assignment, pooling, or repositioning of idle
  vehicles.
- **Charger contention is first-come.** Stalls are reserved on request; there is
  no fleet-wide optimisation of who charges where.
- **Operators always resolve in one tick.** Real teleop has a queue and a
  handling time distribution.
