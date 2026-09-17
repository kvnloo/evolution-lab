from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolution_lab.bench import run_fly_bench
from evolution_lab.select import load_config


class AutoresearchBenchSmoke(unittest.TestCase):
    def test_fly_bench_runs_fast_smoke(self):
        cfg = load_config()
        cfg = json.loads(json.dumps(cfg))
        cfg["bench"]["advise_warm_iters"] = 20
        cfg["bench"]["closed_loop_seeds"] = 8
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fly_bench(config=cfg, run_dir=Path(tmp), skip_unit_tests=True)
            self.assertIsNotNone(result.experiment_id)
            self.assertGreater(result.wall_s, 0.0)
            if result.metrics.error:
                self.fail(result.metrics.error)


if __name__ == "__main__":
    unittest.main()
