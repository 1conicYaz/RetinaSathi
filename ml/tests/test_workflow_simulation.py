from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "run_workflow_simulation.py"
SPEC = importlib.util.spec_from_file_location("workflow_simulation", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class WorkflowSimulationTests(unittest.TestCase):
    def test_all_required_scenarios_exist(self) -> None:
        scenarios = module.scenario_parameters()
        self.assertEqual(len(scenarios), 7)
        self.assertEqual(len({item.name for item in scenarios}), 7)

    def test_added_camera_changes_high_load_outcome(self) -> None:
        scenarios = module.scenario_parameters()
        high_load = module.simulate(scenarios[2])
        added_camera = module.simulate(scenarios[3])
        self.assertGreater(added_camera["patients_processed_per_day"], high_load["patients_processed_per_day"])

    def test_district_configuration_exceeds_100k_annual(self) -> None:
        district = module.simulate(module.scenario_parameters()[-1])
        self.assertGreaterEqual(district["annual_throughput"], 100_000)


if __name__ == "__main__":
    unittest.main()
