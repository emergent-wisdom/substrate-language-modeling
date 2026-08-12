"""Verify the promoted TR-C result files and recompute their reported metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_PATH = (
    Path(__file__).resolve().parent
    / "published"
    / "tr_c_data5x_seed42_t128_release.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(relative_path: str, record: dict) -> Path:
    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {relative_path}")
    actual_size = path.stat().st_size
    if actual_size != record["bytes"]:
        raise RuntimeError(
            f"size mismatch for {relative_path}: "
            f"expected {record['bytes']}, got {actual_size}"
        )
    actual_hash = sha256(path)
    if actual_hash != record["sha256"]:
        raise RuntimeError(
            f"SHA-256 mismatch for {relative_path}: "
            f"expected {record['sha256']}, got {actual_hash}"
        )
    return path


def metrics_from_predictions(path: Path) -> dict[str, float | int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise RuntimeError(f"no predictions in {path}")
    indices = [row["eval_index"] for row in rows]
    if indices != list(range(len(rows))):
        raise RuntimeError(f"non-contiguous eval_index sequence in {path}")

    def accuracy(k: int) -> float:
        correct = sum(
            row["target_id"] in row["top10_ids"][:k]
            for row in rows
        )
        return correct / len(rows)

    loss = math.fsum(-row["target_log_probability"] for row in rows) / len(rows)
    return {
        "eval_loss": loss,
        "eval_perplexity": math.exp(loss),
        "eval_top1": accuracy(1),
        "eval_top5": accuracy(5),
        "eval_top10": accuracy(10),
        "evaluation_examples": len(rows),
    }


def assert_metrics(label: str, actual: dict, expected: dict) -> None:
    for key, actual_value in actual.items():
        expected_value = expected[key]
        if key == "eval_loss":
            tolerance = 1e-7
        elif key == "eval_perplexity":
            tolerance = 1e-6
        else:
            tolerance = 0.0
        if abs(actual_value - expected_value) > tolerance:
            raise RuntimeError(
                f"{label} {key} mismatch: expected {expected_value!r}, "
                f"got {actual_value!r}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-checkpoints",
        action="store_true",
        help="fail unless both large TR-C checkpoints are present and hash-correct",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))

    tracked_paths = {}
    for relative_path, record in release["tracked_artifacts"].items():
        tracked_paths[relative_path] = verify_artifact(relative_path, record)

    for medium in ("description", "pixel"):
        prediction_suffix = f"results/{medium}_tr_c_eval_predictions.jsonl"
        prediction_path = next(
            path for relative, path in tracked_paths.items()
            if relative.endswith(prediction_suffix)
        )
        actual = metrics_from_predictions(prediction_path)
        assert_metrics(
            f"{medium} TR-C",
            actual,
            release["results"][f"{medium}_tr_c"],
        )
        print(f"{medium} TR-C: PASS {json.dumps(actual, sort_keys=True)}")

    missing_checkpoints = []
    for relative_path, record in release["checkpoint_artifacts"].items():
        path = REPO_ROOT / relative_path
        if path.is_file():
            verify_artifact(relative_path, record)
            print(f"checkpoint hash: PASS {relative_path}")
        else:
            missing_checkpoints.append(relative_path)

    if missing_checkpoints and args.require_checkpoints:
        raise RuntimeError(
            "missing checkpoint artifacts: " + ", ".join(missing_checkpoints)
        )
    if missing_checkpoints:
        print(
            "checkpoint replay not checked; archive files are absent: "
            + ", ".join(missing_checkpoints)
        )


if __name__ == "__main__":
    main()
