"""Collect the fixed five-condition STLM comparison into one JSON artifact."""
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
            f"{path} is missing; complete both TR-H trainings first")
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
    if metrics["evaluation_examples"] != config.NN2_EVAL_TARGETS:
        raise RuntimeError(
            f"{label} evaluated {metrics['evaluation_examples']} examples, "
            f"expected {config.NN2_EVAL_TARGETS}")


def trace_metrics(medium):
    path = config.RESULTS / f"{medium}_tr_h.jsonl"
    predictions_path = (
        config.RESULTS / f"{medium}_tr_h_eval_predictions.jsonl")
    metrics = latest_jsonl(path)
    require_metrics(f"{medium} TR-H", metrics)
    if metrics.get("run_id") != config.RUN_ID:
        raise RuntimeError(
            f"{path.name} belongs to run {metrics.get('run_id')!r}, not "
            f"{config.RUN_ID!r}")
    if metrics.get("source_run_id") != config.SOURCE_RUN_ID:
        raise RuntimeError(
            f"{path.name} used source {metrics.get('source_run_id')!r}, not "
            f"{config.SOURCE_RUN_ID!r}")
    if metrics.get("trace_length") != config.TRACE_LENGTH:
        raise RuntimeError(
            f"{path.name} used trace length "
            f"{metrics.get('trace_length')!r}, not {config.TRACE_LENGTH}")
    if metrics.get("history_states") != config.TRACE_LENGTH - 1:
        raise RuntimeError(
            f"{path.name} recorded {metrics.get('history_states')!r} "
            f"history states, not {config.TRACE_LENGTH - 1}")
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"{predictions_path} is missing; TR-H evaluation is incomplete")
    return metrics, path, predictions_path


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
    config.require_experiment_type("trace_readout")
    manifest_path = config.RUN_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} is missing; run write_run_manifest.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = manifest.get("experiment", {})
    if experiment.get("run_id") != config.RUN_ID:
        raise RuntimeError("derived manifest run ID does not match")
    if experiment.get("source_run_id") != config.SOURCE_RUN_ID:
        raise RuntimeError("derived manifest source run ID does not match")
    if (experiment.get("direct_baseline_run_id")
            != config.DIRECT_BASELINE_RUN_ID):
        raise RuntimeError("derived manifest direct baseline run ID does not match")
    if experiment.get("trace_length") != config.TRACE_LENGTH:
        raise RuntimeError("derived manifest trace length does not match")
    if experiment.get("history_states") != config.TRACE_LENGTH - 1:
        raise RuntimeError("derived manifest history-state count does not match")

    published = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))
    source = published["runs"].get(config.SOURCE_RUN_ID)
    if source is None:
        raise RuntimeError(
            f"source run {config.SOURCE_RUN_ID!r} is not published")
    baseline = published["runs"][config.DIRECT_BASELINE_RUN_ID]

    pixel_tr_h, pixel_metrics_path, pixel_predictions_path = trace_metrics(
        "pixel")
    (description_tr_h,
     description_metrics_path,
     description_predictions_path) = trace_metrics("description")

    published_conditions = (
        ("pixel TR-S", source["metrics"]["pixel_nn2"]),
        ("description TR-S", source["metrics"]["text_nn2"]),
        ("direct-token", baseline["metrics"]["direct_token"]),
    )
    for label, metrics in published_conditions:
        require_metrics(label, metrics)

    expected_frequencies = {
        key: source["metrics"]["pixel_nn2"][key]
        for key in FREQUENCY_METRICS
    }
    all_metrics = [
        source["metrics"]["text_nn2"],
        baseline["metrics"]["direct_token"],
        pixel_tr_h,
        description_tr_h,
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
                    "sp": source["checkpoint_sha256"][
                        "nn1_pixel_predictor.pt"],
                    "tr_s": source["checkpoint_sha256"][
                        "nn2_pixel_translator.pt"],
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
                    "sp": source["checkpoint_sha256"][
                        "nn1_text_token_predictor.pt"],
                    "tr_s": source["checkpoint_sha256"][
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
        "pixel_tr_h": condition(
            "pixel",
            "TR-H",
            config.RUN_ID,
            pixel_tr_h,
            {
                "kind": "derived",
                "metrics_path": str(
                    pixel_metrics_path.relative_to(config.RUN_DIR)),
                "metrics_sha256": file_sha256(pixel_metrics_path),
                "predictions_path": str(
                    pixel_predictions_path.relative_to(config.RUN_DIR)),
                "predictions_sha256": file_sha256(pixel_predictions_path),
                "checkpoint_path": str(
                    config.TRACE_PIXEL_CKPT.relative_to(config.RUN_DIR)),
                "checkpoint_sha256": file_sha256(config.TRACE_PIXEL_CKPT),
            },
        ),
        "description_tr_h": condition(
            "description",
            "TR-H",
            config.RUN_ID,
            description_tr_h,
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
                    config.TRACE_DESCRIPTION_CKPT.relative_to(config.RUN_DIR)),
                "checkpoint_sha256": file_sha256(
                    config.TRACE_DESCRIPTION_CKPT),
            },
        ),
    }
    output = {
        "schema_version": 1,
        "comparison": "single-state versus history-conditioned token readout",
        "derived_run_id": config.RUN_ID,
        "source_run_id": config.SOURCE_RUN_ID,
        "baseline_run_id": config.DIRECT_BASELINE_RUN_ID,
        "trace_length": config.TRACE_LENGTH,
        "history_states": config.TRACE_LENGTH - 1,
        "terminology": {
            "SP": "substrate predictor",
            "TR-S": "single-state token readout",
            "TR-H": "history-conditioned token readout",
        },
        "holdout": {
            "method": "fixed evenly spaced next-token contexts",
            "evaluation_examples": config.NN2_EVAL_TARGETS,
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
    print(f"wrote five-condition comparison: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
