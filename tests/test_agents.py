import unittest

from fleetagents import Orchestrator, Point, Policy, ToolRegistry, Trip, Vehicle, World
from fleetagents.agents import Action, EnergyAgent, IntentAgent, Proposal
from fleetagents.domain import Intent, TripState, VehicleState


def make_trip(origin, dest, intent=Intent.CHEAPEST, promised=30.0):
    return Trip(id="T-001", origin=origin, destination=dest, intent=intent,
                promised_minutes=promised, state=TripState.ENROUTE)


class WorldTests(unittest.TestCase):
    def test_same_seed_is_deterministic(self):
        a, b = World(seed=11), World(seed=11)
        for _ in range(30):
            a.tick(2.0)
            b.tick(2.0)
        self.assertEqual([i.id for i in a.incidents], [i.id for i in b.incidents])
        self.assertAlmostEqual(a.congestion_at(Point(5, 5)), b.congestion_at(Point(5, 5)))

    def test_weather_degrades_speed(self):
        w = World(seed=3)
        under_storm = w.weather[0].center
        clear = Point(w.size_km - 1, w.size_km - 1)
        self.assertGreater(w.weather_at(under_storm), w.weather_at(clear))


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=5)
        self.tools = ToolRegistry(self.world)

    def test_permission_is_enforced(self):
        res = self.tools.call("route", "reserve_charger", charger_id="CHG-DEPOT", vehicle_id="AV-1")
        self.assertFalse(res.ok)
        self.assertIn("not permitted", res.error)

    def test_reserve_and_release_stalls(self):
        before = self.world.charger("CHG-DEPOT").available
        self.tools.call("energy", "reserve_charger", charger_id="CHG-DEPOT", vehicle_id="AV-1")
        self.assertEqual(self.world.charger("CHG-DEPOT").available, before - 1)
        self.tools.call("energy", "release_charger", charger_id="CHG-DEPOT")
        self.assertEqual(self.world.charger("CHG-DEPOT").available, before)

    def test_every_call_is_audited(self):
        self.tools.call("route", "get_traffic", point=Point(4, 4))
        self.assertEqual(self.tools.calls[-1]["tool"], "get_traffic")
        self.assertTrue(self.tools.calls[-1]["ok"])


class EnergyAgentTests(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=5)
        self.agent = EnergyAgent(ToolRegistry(self.world), self.world)

    def test_low_soc_forces_a_hard_charge_stop(self):
        v = Vehicle(id="AV-1", position=Point(2, 2), soc=0.12, state=VehicleState.ENROUTE)
        v.trip = make_trip(v.position, Point(20, 20))
        charge = [p for p in self.agent.propose(v, {}) if p.action is Action.CHARGE_STOP]
        self.assertTrue(charge, "expected a charge stop proposal")
        self.assertTrue(charge[0].hard_constraint)

    def test_healthy_soc_continues(self):
        v = Vehicle(id="AV-2", position=Point(10, 10), soc=0.9, state=VehicleState.ENROUTE)
        v.trip = make_trip(v.position, Point(13, 13))
        actions = {p.action for p in self.agent.propose(v, {})}
        self.assertNotIn(Action.CHARGE_STOP, actions)


class IntentAgentTests(unittest.TestCase):
    def test_detour_budget_reflects_intent(self):
        world = World(seed=5)
        agent = IntentAgent(ToolRegistry(world), world)
        v = Vehicle(id="AV-3", position=Point(5, 5), soc=0.6, state=VehicleState.ENROUTE)
        v.trip = make_trip(v.position, Point(15, 15), intent=Intent.FASTEST)
        budget = agent.propose(v, {})[0].payload["detour_budget_min"]
        self.assertAlmostEqual(budget, Intent.FASTEST.max_detour_min)


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=5)
        self.orch = Orchestrator(self.world, policy=Policy(min_confidence=0.45))

    def test_hard_constraint_beats_higher_scoring_soft_proposal(self):
        soft = Proposal("route", Action.REROUTE, "saves time", utility=99.0, confidence=1.0)
        hard = Proposal("energy", Action.CHARGE_STOP, "below reserve", utility=1.0,
                        confidence=0.9, hard_constraint=True)
        self.assertIs(self.orch._arbitrate([soft, hard]), hard)

    def test_low_confidence_proposal_is_vetoed(self):
        v = Vehicle(id="AV-9", position=Point(5, 5), soc=0.8, state=VehicleState.ENROUTE)
        v.trip = make_trip(v.position, Point(12, 12))
        timid = Proposal("route", Action.REROUTE, "not sure", utility=5.0, confidence=0.2)
        kept, vetoed = self.orch._guardrails(v, [timid])
        self.assertEqual(kept, [])
        self.assertIn("confidence", vetoed[0]["reason"])

    def test_reroute_limit_is_enforced(self):
        v = Vehicle(id="AV-8", position=Point(5, 5), soc=0.8, state=VehicleState.ENROUTE)
        v.trip = make_trip(v.position, Point(12, 12))
        self.orch.reroute_counts[v.id] = self.orch.policy.max_reroutes_per_trip
        p = Proposal("route", Action.REROUTE, "again", utility=5.0, confidence=0.9,
                     payload={"waypoint": Point(6, 6), "gain_min": 5.0})
        kept, vetoed = self.orch._guardrails(v, [p])
        self.assertEqual(kept, [])
        self.assertIn("reroute limit", vetoed[0]["reason"])

    def test_handoff_pauses_vehicle_and_opens_ticket(self):
        v = Vehicle(id="AV-7", position=Point(5, 5), soc=0.5, state=VehicleState.ENROUTE)
        v.trip = make_trip(v.position, Point(12, 12))
        p = Proposal("assist", Action.HANDOFF, "blocked", utility=8.0, confidence=0.9,
                     payload={"reason": "blocked", "severity": "high"})
        self.orch._execute(v, p)
        self.assertIs(v.state, VehicleState.AWAITING_OPERATOR)
        self.assertTrue(v.operator_ticket.startswith("RA-"))
        self.assertEqual(self.orch.stats["handoffs"], 1)


class SimulationTests(unittest.TestCase):
    def _run(self, **kw):
        from fleetagents import build_scenario
        from fleetagents.telemetry import Trace
        trace = Trace()
        scenario = build_scenario(**kw)
        orch = Orchestrator(scenario.world, trace=trace)
        for _ in range(70):
            active = [v for v in scenario.vehicles
                      if v.trip and v.trip.state is not TripState.COMPLETED]
            if not active:
                break
            orch.step(active, minutes=2.0)
        return scenario, orch, trace

    def test_full_run_completes_and_traces(self):
        from fleetagents import kpis
        scenario, orch, trace = self._run(n_vehicles=6, seed=21)
        results = kpis(scenario.vehicles, orch.stats)
        self.assertGreater(results["completed"], 0)
        self.assertTrue(trace.of_kind("decision"))

    def test_no_vehicle_strands_at_zero_charge(self):
        scenario, _, _ = self._run(n_vehicles=8, seed=4, low_battery_share=0.9)
        self.assertTrue(all(v.soc > 0.0 for v in scenario.vehicles))


if __name__ == "__main__":
    unittest.main()


class RouteAgentTests(unittest.TestCase):
    def test_blocking_incident_on_the_path_triggers_a_reroute(self):
        from fleetagents.agents import RouteAgent
        from fleetagents.world import Incident

        world = World(seed=5)
        world.weather = []          # isolate the traffic signal
        world.incidents = [
            Incident(id="INC-TEST", location=Point(10.0, 10.0), radius_km=2.5,
                     severity=0.95, expires_at=1e9, description="collision blocking two lanes")
        ]
        agent = RouteAgent(ToolRegistry(world), world)
        v = Vehicle(id="AV-R", position=Point(4.0, 4.0), soc=0.8, state=VehicleState.ENROUTE)
        v.trip = make_trip(v.position, Point(16.0, 16.0))   # straight through the incident

        actions = {p.action for p in agent.propose(v, {})}
        self.assertIn(Action.REROUTE, actions,
                      "an incident squarely on the path should produce a reroute")
