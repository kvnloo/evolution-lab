from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evolution_lab.advise import _local_plasticity_student, clear_student_cache
from evolution_lab.engine import seed_genomes
from evolution_lab.models import fit_student
from evolution_lab.splits import load_splits
from evolution_lab.student_bundle import load_bundle, save_bundle


class StudentBundleTests(unittest.TestCase):
    def test_save_load_roundtrip(self):
        genome = next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
        student = fit_student(genome, load_splits().train)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "student.npz"
            save_bundle(genome, student, path, metrics={"closed_loop": 1.0})
            loaded = load_bundle(genome, path)
            sample = load_splits().confirm[:4]
            self.assertTrue(
                (student.predict_fn(sample) == loaded.predict_fn(sample)).all()
            )

    def test_advise_prefers_bundle_when_present(self):
        genome = next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
        student = fit_student(genome, load_splits().train)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "recovery_student.npz"
            save_bundle(genome, student, bundle)
            clear_student_cache()
            import os

            os.environ["FLYFORGE_RECOVERY_BUNDLE"] = str(bundle)
            try:
                _, loaded = _local_plasticity_student()
                self.assertEqual(loaded.extras.get("bundle"), str(bundle))
            finally:
                clear_student_cache()
                os.environ.pop("FLYFORGE_RECOVERY_BUNDLE", None)


    def test_locked_split_closed_loop_is_perfect(self):
        from evolution_lab.dagger import mean_closed_loop_reward, make_env
        from evolution_lab.splits import load_splits

        genome = next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
        student = fit_student(genome, load_splits().train)
        env = make_env("hermes_recovery", delayed_cue=True, history=genome.architecture.history)
        mean = mean_closed_loop_reward(env, student, list(range(24)))
        self.assertGreaterEqual(mean, 1.0)

if __name__ == "__main__":
    unittest.main()
