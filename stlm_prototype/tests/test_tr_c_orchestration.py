"""Focused checks for the reproducible TR-C orchestration artifacts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))

import config  # noqa: E402
import collect_tr_c_five_condition_results as collector  # noqa: E402
import write_run_manifest  # noqa: E402


class ContextualOrchestrationTests(unittest.TestCase):

    def test_manifest_distinguishes_windows_from_supervised_positions(self):
        with (
            patch.object(config, "EXPERIMENT_TYPE", "contextual_readout"),
            patch.object(config, "CONTEXTUAL_TRAIN_WINDOWS", 100_000),
            patch.object(config, "NN2_EVAL_TARGETS", 1_024),
            patch.object(config, "NN1_CONTEXT", 128),
        ):
            experiment = write_run_manifest.base_experiment()

        self.assertEqual(experiment["training_windows"], 100_000)
        self.assertEqual(
            experiment["supervised_target_positions"], 12_800_000)
        self.assertEqual(
            experiment["evaluation"]["final_position_contexts"], 1_024)
        self.assertEqual(
            experiment["causal_token_readout"]["loss"],
            "cross-entropy over every aligned position",
        )
        self.assertEqual(
            experiment["initialization"],
            "fresh deterministic initialization",
        )

    def test_collector_requires_dense_tr_c_result_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            metrics = {
                "run_id": "derived",
                "source_run_id": "source",
                "medium": "pixel",
                "readout": "TR-C",
                "objective": "all_aligned_positions",
                "context_length": 4,
                "training_windows": 3,
                "supervised_target_positions": 12,
                "optimizer_updates": 2,
                "source_checkpoint_sha256": {"sp": "abc"},
                "eval_loss": 1.0,
                "eval_top1": 0.5,
                "eval_top5": 1.0,
                "eval_top10": 1.0,
                "evaluation_examples": 2,
            }
            (results / "pixel_tr_c.jsonl").write_text(
                json.dumps(metrics) + "\n", encoding="utf-8")
            predictions = results / "pixel_tr_c_eval_predictions.jsonl"
            predictions.write_text("{}\n{}\n", encoding="utf-8")

            with (
                patch.object(config, "RESULTS", results),
                patch.object(config, "RUN_ID", "derived"),
                patch.object(config, "SOURCE_RUN_ID", "source"),
                patch.object(config, "READOUT_EVAL_TARGETS", 2),
                patch.object(config, "CONTEXTUAL_TRAIN_WINDOWS", 3),
                patch.object(config, "NN1_CONTEXT", 4),
                patch.object(config, "CONTEXTUAL_PIXEL_BATCH", 2),
            ):
                actual, metrics_path, predictions_path = (
                    collector.contextual_metrics("pixel", "abc"))

            self.assertEqual(actual, metrics)
            self.assertEqual(metrics_path, results / "pixel_tr_c.jsonl")
            self.assertEqual(predictions_path, predictions)

    def test_manifest_rejects_different_media_window_sequences(self):
        with (
            patch.object(config, "EXPERIMENT_TYPE", "contextual_readout"),
            patch.object(config, "CONTEXTUAL_PIXEL_BATCH", 8),
            patch.object(config, "CONTEXTUAL_DESCRIPTION_BATCH", 4),
        ):
            with self.assertRaisesRegex(RuntimeError, "same training windows"):
                write_run_manifest.base_experiment()


if __name__ == "__main__":
    unittest.main()
