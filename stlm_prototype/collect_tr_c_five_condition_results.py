"""Collect the fixed five-condition TR-S/TR-C comparison as one artifact."""

import json
import math

import config
from training_utils import file_sha256


PUBLISHED_PATH = config.ROOT / "published" / "pilot_runs.json"
OUTPUT_PATH = config.RESULTS / "five_condition_comparison.json"
REQUIRED_METRICS = (
    "eval_loss",
    "eval_top1",
    "eval_top5",
    "eval_top10",
    "evaluation_examples",
)
FREQUENCY_METRICS = (
    "frequency_top1",
    "frequency_top5",
    "frequency_top10",
)


def latest_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; complete both TR-C trainings first")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"{path} contains no metric records")
    return records[-1]


def require_metrics(label, metrics):
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise RuntimeError(
            f"{label} metrics are incomplete; missing {', '.join(missing)}")
    if metrics["evaluation_examples"] != config.READOUT_EVAL_TARGETS:
        raise RuntimeError(
            f"{label} evaluated {metrics['evaluation_examples']} examples, "
            f"expected {config.READOUT_EVAL_TARGETS}")


def require_prediction_count(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; {label} evaluation is incomplete")
    count = sum(
        bool(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    if count != config.READOUT_EVAL_TARGETS:
        raise RuntimeError(
            f"{label} has {count} recorded predictions, expected "
            f"{config.READOUT_EVAL_TARGETS}")


def contextual_metrics(medium, expected_sp_hash):
    metrics_path = config.RESULTS / f"{medium}_tr_c.jsonl"
    predictions_path = (
        config.RESULTS / f"{medium}_tr_c_eval_predictions.jsonl")
    metrics = latest_jsonl(metrics_path)
    label = f"{medium} TR-C"
    require_metrics(label, metrics)
    expected = {
        "run_id": config.RUN_ID,
        "source_run_id": config.SOURCE_RUN_ID,
        "medium": medium,
        "readout": "TR-C",
        "objective": "all_aligned_positions",
        "context_length": config.NN1_CONTEXT,
        "training_windows": config.CONTEXTUAL_TRAIN_WINDOWS,
        "supervised_target_positions": (
            config.CONTEXTUAL_TRAIN_WINDOWS * config.NN1_CONTEXT),
        "optimizer_updates": math.ceil(
            config.CONTEXTUAL_TRAIN_WINDOWS
            / (config.CONTEXTUAL_PIXEL_BATCH
               if medium == "pixel"
               else config.CONTEXTUAL_DESCRIPTION_BATCH)),
        "source_checkpoint_sha256": {"sp": expected_sp_hash},
    }
    mismatches = {
        key: (value, metrics.get(key))
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        details = "; ".join(
            f"{key}: expected {want!r}, got {got!r}"
            for key, (want, got) in sorted(mismatches.items()))
        raise RuntimeError(f"{label} provenance mismatch ({details})")
    require_prediction_count(predictions_path, label)
    return metrics, metrics_path, predictions_path


def condition(medium, readout, run_id, metrics, provenance):
    comparable_metrics = dict(metrics)
    comparable_metrics.setdefault(
        "eval_perplexity", math.exp(comparable_metrics["eval_loss"]))
    return {
        "medium": medium,
        "readout": readout,
        "run_id": run_id,
        "metrics": comparable_metrics,
        "provenance": provenance,
    }


def main():
    config.require_experiment_type("contextual_readout")
    manifest_path = config.RUN_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} is missing; run write_run_manifest.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = manifest.get("experiment", {})
    expected_manifest = {
        "run_id": config.RUN_ID,
        "experiment_type": "contextual_readout",
        "source_run_id": config.SOURCE_RUN_ID,
        "direct_baseline_run_id": config.DIRECT_BASELINE_RUN_ID,
        "state_sequence_length": config.NN1_CONTEXT,
        "training_windows": config.CONTEXTUAL_TRAIN_WINDOWS,
        "supervised_target_positions": (
            config.CONTEXTUAL_TRAIN_WINDOWS * config.NN1_CONTEXT),
        "initialization": "fresh deterministic initialization",
        "shared_training_windows_across_media": True,
    }
    mismatches = {
        key: (value, experiment.get(key))
        for key, value in expected_manifest.items()
        if experiment.get(key) != value
    }
    if mismatches:
        details = "; ".join(
            f"{key}: expected {want!r}, got {got!r}"
            for key, (want, got) in sorted(mismatches.items()))
        raise RuntimeError(f"derived manifest mismatch ({details})")

    published = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))
    source = published["runs"].get(config.SOURCE_RUN_ID)
    if source is None:
        raise RuntimeError(
            f"source run {config.SOURCE_RUN_ID!r} is not published")
    baseline = published["runs"].get(config.DIRECT_BASELINE_RUN_ID)
    if baseline is None or baseline.get("experiment_type") != "direct_token":
        raise RuntimeError(
            f"baseline run {config.DIRECT_BASELINE_RUN_ID!r} is not a "
            "published direct-token run")

    source_hashes = source["checkpoint_sha256"]
    pixel_tr_c, pixel_metrics_path, pixel_predictions_path = (
        contextual_metrics(
            "pixel", source_hashes["nn1_pixel_predictor.pt"]))
    (description_tr_c,
     description_metrics_path,
     description_predictions_path) = contextual_metrics(
         "description", source_hashes["nn1_text_token_predictor.pt"])

    published_conditions = (
        ("pixel TR-S", source["metrics"]["pixel_nn2"]),
        ("description TR-S", source["metrics"]["text_nn2"]),
        ("direct-token", baseline["metrics"]["direct_token"]),
    )
    for label, metrics in published_conditions:
        require_metrics(label, metrics)
    tr_s_training_counts = {
        source["metrics"]["pixel_nn2"].get("training_examples"),
        source["metrics"]["text_nn2"].get("training_examples"),
    }
    if len(tr_s_training_counts) != 1 or None in tr_s_training_counts:
        raise RuntimeError(
            "published TR-S conditions do not record one shared training "
            "target count")
    published_tr_s_training_targets = tr_s_training_counts.pop()

    expected_frequencies = {
        key: source["metrics"]["pixel_nn2"][key]
        for key in FREQUENCY_METRICS
    }
    all_metrics = [
        source["metrics"]["text_nn2"],
        baseline["metrics"]["direct_token"],
        pixel_tr_c,
        description_tr_c,
    ]
    for metrics in all_metrics:
        for key, expected in expected_frequencies.items():
            if metrics.get(key) != expected:
                raise RuntimeError(
                    f"holdout mismatch for {key}: expected {expected!r}, "
                    f"got {metrics.get(key)!r}")

    source_registry = str(PUBLISHED_PATH.relative_to(config.ROOT))
    conditions = {
        "pixel_tr_s": condition(
            "pixel",
            "TR-S",
            config.SOURCE_RUN_ID,
            source["metrics"]["pixel_nn2"],
            {
                "kind": "published",
                "registry": source_registry,
                "checkpoint_sha256": {
                    "sp": source_hashes["nn1_pixel_predictor.pt"],
                    "tr_s": source_hashes["nn2_pixel_translator.pt"],
                },
            },
        ),
        "description_tr_s": condition(
            "description",
            "TR-S",
            config.SOURCE_RUN_ID,
            source["metrics"]["text_nn2"],
            {
                "kind": "published",
                "registry": source_registry,
                "checkpoint_sha256": {
                    "sp": source_hashes["nn1_text_token_predictor.pt"],
                    "tr_s": source_hashes[
                        "nn2_text_transformer_translator.pt"],
                },
            },
        ),
        "direct_token": condition(
            "tokens",
            "direct",
            config.DIRECT_BASELINE_RUN_ID,
            baseline["metrics"]["direct_token"],
            {
                "kind": "published",
                "registry": source_registry,
                "checkpoint_sha256": baseline["checkpoint_sha256"][
                    "baseline_direct_token.pt"],
            },
        ),
        "pixel_tr_c": condition(
            "pixel",
            "TR-C",
            config.RUN_ID,
            pixel_tr_c,
            {
                "kind": "derived",
                "metrics_path": str(
                    pixel_metrics_path.relative_to(config.RUN_DIR)),
                "metrics_sha256": file_sha256(pixel_metrics_path),
                "predictions_path": str(
                    pixel_predictions_path.relative_to(config.RUN_DIR)),
                "predictions_sha256": file_sha256(pixel_predictions_path),
                "checkpoint_path": str(
                    config.CONTEXTUAL_PIXEL_CKPT.relative_to(config.RUN_DIR)),
                "checkpoint_sha256": file_sha256(
                    config.CONTEXTUAL_PIXEL_CKPT),
            },
        ),
        "description_tr_c": condition(
            "description",
            "TR-C",
            config.RUN_ID,
            description_tr_c,
            {
                "kind": "derived",
                "metrics_path": str(
                    description_metrics_path.relative_to(config.RUN_DIR)),
                "metrics_sha256": file_sha256(description_metrics_path),
                "predictions_path": str(
                    description_predictions_path.relative_to(config.RUN_DIR)),
                "predictions_sha256": file_sha256(
                    description_predictions_path),
                "checkpoint_path": str(
                    config.CONTEXTUAL_DESCRIPTION_CKPT.relative_to(
                        config.RUN_DIR)),
                "checkpoint_sha256": file_sha256(
                    config.CONTEXTUAL_DESCRIPTION_CKPT),
            },
        ),
    }
    output = {
        "schema_version": 1,
        "comparison": "single-state versus causal-sequence token readout",
        "comparison_caveat": (
            f"TR-C is trained on "
            f"{config.CONTEXTUAL_TRAIN_WINDOWS * config.NN1_CONTEXT:,} "
            "supervised target positions; the published TR-S conditions "
            f"used {published_tr_s_training_targets:,} "
            "single-position targets. This is not a supervision-matched "
            "history-effect estimate."
        ),
        "derived_run_id": config.RUN_ID,
        "source_run_id": config.SOURCE_RUN_ID,
        "baseline_run_id": config.DIRECT_BASELINE_RUN_ID,
        "terminology": {
            "SP": "substrate predictor",
            "TR-S": "single-state token readout",
            "TR-C": "causal token readout",
        },
        "training_protocol": {
            "source_sp": "frozen published checkpoint",
            "tr_c_initialization": "fresh deterministic initialization",
            "shared_training_windows_across_media": True,
            "state_sequence_length": config.NN1_CONTEXT,
            "state_target_alignment":
                "same index: SP state t predicts source token t+1",
            "objective": "cross-entropy over every aligned position",
            "training_windows": config.CONTEXTUAL_TRAIN_WINDOWS,
            "supervised_target_positions": (
                config.CONTEXTUAL_TRAIN_WINDOWS * config.NN1_CONTEXT),
        },
        "holdout": {
            "method": "fixed evenly spaced next-token contexts",
            "position": "final state in each context",
            "evaluation_examples": config.READOUT_EVAL_TARGETS,
            **expected_frequencies,
        },
        "conditions": conditions,
        "provenance": {
            "derived_manifest": str(
                manifest_path.relative_to(config.RUN_DIR)),
            "derived_manifest_sha256": file_sha256(manifest_path),
            "published_registry": source_registry,
            "published_registry_sha256": file_sha256(PUBLISHED_PATH),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(OUTPUT_PATH)
    print(f"wrote five-condition TR-C comparison: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
