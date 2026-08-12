"""Create an immutable manifest for a resumable experiment run."""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import torch

import config
from corpus import load_corpus
from training_utils import verify_file_sha256
from vocab import Tokenizer

PUBLISHED_RUNS_PATH = config.ROOT / "published" / "pilot_runs.json"


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=config.ROOT.parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def base_experiment():
    experiment = {
        "run_id": config.RUN_ID,
        "seed": config.SEED,
        "corpus": config.CORPUS_NAME,
        "max_sentences": config.MAX_SENTENCES,
        "epochs": {
            "text_nn1": config.NN1_TEXT_EPOCHS,
            "pixel_nn1": config.NN1_EPOCHS,
        },
        "translator_train_targets": config.NN2_TRAIN_TARGETS,
        "batch_sizes": {
            "text_nn1": config.NN1_TEXT_BATCH,
            "text_nn2": config.NN2_TEXT_BATCH,
            "pixel_nn1": config.NN1_BATCH,
            "pixel_nn2": config.NN2_BATCH,
        },
        "context": config.NN1_CONTEXT,
        "evaluation": {
            "holdout_tokens": config.NN1_CONTEXT * config.EVAL_CONTEXTS,
            "translator_targets": config.NN2_EVAL_TARGETS,
            "method": "fixed evenly spaced next-token contexts",
        },
        "vocab_size": config.VOCAB_SIZE,
        "image_size": config.IMG_SIZE,
        "text_description_length": config.TEXT_DESC_LEN,
        "text_dimension": config.TEXT_DIM,
    }
    if config.EXPERIMENT_TYPE == "trace_readout":
        experiment.pop("epochs")
        experiment["readout_train_targets"] = experiment.pop(
            "translator_train_targets")
        experiment["evaluation"]["readout_targets"] = (
            experiment["evaluation"].pop("translator_targets"))
        experiment["batch_sizes"] = {
            "pixel_tr_h": config.TRACE_PIXEL_BATCH,
            "description_tr_h": config.TRACE_DESCRIPTION_BATCH,
        }
        experiment.update({
            "source_run_id": config.SOURCE_RUN_ID,
            "direct_baseline_run_id": config.DIRECT_BASELINE_RUN_ID,
            "trace_length": config.TRACE_LENGTH,
            "history_states": config.TRACE_LENGTH - 1,
            "trace_semantics":
                "matched causal window: prior states plus current state",
            "readout_scope": "causal substrate trace only",
            "source_token_access": False,
            "rng_seeds": {
                "adapter_initialization":
                    config.TRACE_ADAPTER_INIT_SEED,
                "context_sampling":
                    config.TRACE_CONTEXT_SAMPLING_SEED,
                "history_order_diagnostic":
                    config.TRACE_DIAGNOSTIC_SHUFFLE_SEED,
            },
            "history_adapter": {
                "architecture":
                    "frozen TR-S plus zero-initialized logit residual",
                "temporal_layer": "one-layer Transformer encoder",
                "optimizer": {
                    "name": "AdamW",
                    "learning_rate": config.TRACE_LR,
                    "betas": [0.9, 0.999],
                    "epsilon": 1e-8,
                    "weight_decay": 0.01,
                },
                "nonfinite_gradient_guard": True,
                "state_dimension": config.TRACE_STATE_DIM,
                "temporal_feedforward_dimension":
                    config.TRACE_TEMPORAL_FF_DIM,
                "temporal_attention_heads":
                    config.TRACE_TEMPORAL_HEADS,
                "pixel_pool_grid": config.TRACE_PIXEL_GRID,
                "description_pool_bins": config.TRACE_DESCRIPTION_BINS,
                "save_every_steps": config.TRACE_SAVE_EVERY_STEPS,
            },
        })
    elif config.EXPERIMENT_TYPE == "contextual_readout":
        if (config.CONTEXTUAL_PIXEL_BATCH
                != config.CONTEXTUAL_DESCRIPTION_BATCH):
            raise RuntimeError(
                "TR-C media must use the same logical batch size so their "
                "seeded samplers produce the same training windows")
        experiment.pop("epochs")
        experiment.pop("translator_train_targets")
        training_windows = config.CONTEXTUAL_TRAIN_WINDOWS
        evaluation_contexts = experiment["evaluation"].pop(
            "translator_targets")
        supervised_target_positions = training_windows * config.NN1_CONTEXT
        experiment["batch_sizes"] = {
            "pixel_tr_c": config.CONTEXTUAL_PIXEL_BATCH,
            "description_tr_c": config.CONTEXTUAL_DESCRIPTION_BATCH,
        }
        experiment["microbatch_sizes"] = {
            "pixel_tr_c": config.CONTEXTUAL_PIXEL_MICROBATCH,
            "description_tr_c": config.CONTEXTUAL_DESCRIPTION_MICROBATCH,
        }
        experiment["evaluation"].update({
            "final_position_contexts": evaluation_contexts,
            "position": "final state in each fixed context",
        })
        experiment.update({
            "source_run_id": config.SOURCE_RUN_ID,
            "direct_baseline_run_id": config.DIRECT_BASELINE_RUN_ID,
            "state_sequence_length": config.NN1_CONTEXT,
            "training_windows": training_windows,
            "supervised_target_positions_per_window": config.NN1_CONTEXT,
            "supervised_target_positions": supervised_target_positions,
            "optimizer_updates": {
                "pixel_tr_c": (
                    training_windows + config.CONTEXTUAL_PIXEL_BATCH - 1
                ) // config.CONTEXTUAL_PIXEL_BATCH,
                "description_tr_c": (
                    training_windows
                    + config.CONTEXTUAL_DESCRIPTION_BATCH - 1
                ) // config.CONTEXTUAL_DESCRIPTION_BATCH,
            },
            "state_target_alignment":
                "same index: SP state t predicts source token t+1",
            "readout_scope": "causal substrate-state sequence only",
            "source_token_access": False,
            "source_sp_frozen": True,
            "shared_training_windows_across_media": True,
            "initialization": "fresh deterministic initialization",
            "rng_seeds": {
                "readout_initialization":
                    config.CONTEXTUAL_READOUT_INIT_SEED,
                "context_sampling":
                    config.CONTEXTUAL_CONTEXT_SAMPLING_SEED,
            },
            "causal_token_readout": {
                "architecture": "pre-norm causal Transformer encoder",
                "loss": "cross-entropy over every aligned position",
                "d_model": config.CONTEXTUAL_D_MODEL,
                "attention_heads": config.CONTEXTUAL_HEADS,
                "layers": config.CONTEXTUAL_LAYERS,
                "feedforward_dimension": config.CONTEXTUAL_FF_DIM,
                "dropout": 0.0,
                "pixel_pool_grid": config.CONTEXTUAL_PIXEL_POOL_GRID,
                "description_pool_bins":
                    config.CONTEXTUAL_DESCRIPTION_POOL_BINS,
                "optimizer": {
                    "name": "AdamW",
                    "learning_rate": config.CONTEXTUAL_LR,
                    "betas": [0.9, 0.999],
                    "epsilon": 1e-8,
                    "weight_decay": 0.01,
                },
                "gradient_clip_norm": None,
                "nonfinite_gradient_guard": True,
                "save_every_steps": config.CONTEXTUAL_SAVE_EVERY_STEPS,
            },
        })
    return experiment


def package_versions():
    result = {}
    for package in (
            "datasets", "diffusers", "numpy", "pillow", "torch",
            "transformers", "tqdm"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def runtime_identity():
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "packages": package_versions(),
        "device": config.DEVICE,
        "platform": platform.platform(),
    }


def verified_inputs():
    inputs = {
        "vocabulary": {
            "path": "data/vocab.json",
            "sha256": verify_file_sha256(
                config.VOCAB_FILE, config.VOCAB_SHA256, "vocabulary"),
        },
    }
    if config.EXPERIMENT_TYPE == "direct_token":
        return inputs
    if config.EXPERIMENT_TYPE == "trace_readout":
        inputs["source_components"] = verified_trace_source()
        inputs["direct_baseline"] = verified_direct_baseline()
        return inputs
    if config.EXPERIMENT_TYPE == "contextual_readout":
        inputs["source_components"] = verified_contextual_source()
        inputs["direct_baseline"] = verified_direct_baseline()
        return inputs
    inputs.update({
        "descriptions": {
            "path": "data/word_symbolic_descriptions.json",
            "sha256": verify_file_sha256(
                config.DESCRIPTIONS,
                config.DESCRIPTIONS_SHA256,
                "symbolic descriptions",
            ),
        },
        "pixel_canonicals": {
            "generator": config.SANA_MODEL_NAME,
            "generator_revision": config.SANA_MODEL_REVISION,
            "path": "data/canonicals.npz",
            "sha256": verify_file_sha256(
                config.CANONICALS_FILE,
                config.PIXEL_CANONICALS_SHA256,
                "pixel canonicals",
            ),
        },
        "text_canonicals": {
            "path": "data/text_token_canonicals.npz",
            "sha256": verify_file_sha256(
                config.TEXT_CANONICALS_FILE,
                config.TEXT_CANONICALS_SHA256,
                "text canonicals",
            ),
            "encoder": config.TEXT_ENCODER_NAME,
            "encoder_revision": config.TEXT_ENCODER_REVISION,
        },
    })
    return inputs


def verified_trace_source():
    if config.RUN_ID == config.SOURCE_RUN_ID:
        raise RuntimeError(
            "a derived trace-readout run cannot write into its source run; "
            "set STLM_RUN_ID to a new identifier")
    published = json.loads(PUBLISHED_RUNS_PATH.read_text(encoding="utf-8"))
    try:
        source = published["runs"][config.SOURCE_RUN_ID]
        recovered_input_identity = published["recovered_input_identity"]
    except KeyError as exc:
        raise RuntimeError(
            f"trace source {config.SOURCE_RUN_ID!r} is not recorded in "
            f"{PUBLISHED_RUNS_PATH.relative_to(config.ROOT)}") from exc
    checkpoint_hashes = source.get("checkpoint_sha256", {})
    required = {
        "pixel_sp": (
            "nn1_pixel_predictor.pt",
            config.SOURCE_PIXEL_SP_CKPT,
        ),
        "pixel_tr_s": (
            "nn2_pixel_translator.pt",
            config.SOURCE_PIXEL_TR_S_CKPT,
        ),
        "description_sp": (
            "nn1_text_token_predictor.pt",
            config.SOURCE_DESCRIPTION_SP_CKPT,
        ),
        "description_tr_s": (
            "nn2_text_transformer_translator.pt",
            config.SOURCE_DESCRIPTION_TR_S_CKPT,
        ),
    }
    verified = {}
    for label, (filename, path) in required.items():
        expected = checkpoint_hashes.get(filename)
        if expected is None:
            raise RuntimeError(
                f"published source {config.SOURCE_RUN_ID!r} has no "
                f"{filename!r} digest")
        verified[label] = {
            "path": str(path.relative_to(config.ROOT)),
            "sha256": verify_file_sha256(
                path,
                expected,
                f"{config.SOURCE_RUN_ID}/{filename}",
            ),
        }
    return {
        "run_id": config.SOURCE_RUN_ID,
        "registry": str(PUBLISHED_RUNS_PATH.relative_to(config.ROOT)),
        "source_git_commit": source.get("git_commit"),
        "recovered_input_identity": recovered_input_identity,
        "checkpoints": verified,
    }


def verified_contextual_source():
    """Verify the frozen SP checkpoints used to construct TR-C states."""
    if config.RUN_ID == config.SOURCE_RUN_ID:
        raise RuntimeError(
            "a derived contextual-readout run cannot write into its source "
            "run; set STLM_RUN_ID to a new identifier")
    published = json.loads(PUBLISHED_RUNS_PATH.read_text(encoding="utf-8"))
    try:
        source = published["runs"][config.SOURCE_RUN_ID]
        recovered_input_identity = published["recovered_input_identity"]
    except KeyError as exc:
        raise RuntimeError(
            f"contextual source {config.SOURCE_RUN_ID!r} is not recorded in "
            f"{PUBLISHED_RUNS_PATH.relative_to(config.ROOT)}") from exc
    checkpoint_hashes = source.get("checkpoint_sha256", {})
    required = {
        "pixel_sp": (
            "nn1_pixel_predictor.pt",
            config.SOURCE_PIXEL_SP_CKPT,
        ),
        "description_sp": (
            "nn1_text_token_predictor.pt",
            config.SOURCE_DESCRIPTION_SP_CKPT,
        ),
    }
    verified = {}
    for label, (filename, path) in required.items():
        expected = checkpoint_hashes.get(filename)
        if expected is None:
            raise RuntimeError(
                f"published source {config.SOURCE_RUN_ID!r} has no "
                f"{filename!r} digest")
        verified[label] = {
            "path": str(path.relative_to(config.ROOT)),
            "sha256": verify_file_sha256(
                path,
                expected,
                f"{config.SOURCE_RUN_ID}/{filename}",
            ),
        }
    return {
        "run_id": config.SOURCE_RUN_ID,
        "registry": str(PUBLISHED_RUNS_PATH.relative_to(config.ROOT)),
        "source_git_commit": source.get("git_commit"),
        "recovered_input_identity": recovered_input_identity,
        "checkpoints": verified,
    }


def verified_direct_baseline():
    published = json.loads(PUBLISHED_RUNS_PATH.read_text(encoding="utf-8"))
    try:
        baseline = published["runs"][config.DIRECT_BASELINE_RUN_ID]
        expected = baseline["checkpoint_sha256"]["baseline_direct_token.pt"]
    except KeyError as exc:
        raise RuntimeError(
            f"direct baseline {config.DIRECT_BASELINE_RUN_ID!r} is not a "
            "published direct-token run") from exc
    if baseline.get("experiment_type") != "direct_token":
        raise RuntimeError(
            f"{config.DIRECT_BASELINE_RUN_ID!r} is not a direct-token run")
    return {
        "run_id": config.DIRECT_BASELINE_RUN_ID,
        "registry": str(PUBLISHED_RUNS_PATH.relative_to(config.ROOT)),
        "checkpoint": {
            "path": str(config.DIRECT_BASELINE_CKPT.relative_to(config.ROOT)),
            "sha256": verify_file_sha256(
                config.DIRECT_BASELINE_CKPT,
                expected,
                f"{config.DIRECT_BASELINE_RUN_ID}/baseline_direct_token.pt",
            ),
        },
    }


def verify_trace_corpus_identity(identity):
    if config.EXPERIMENT_TYPE not in {
            "trace_readout", "contextual_readout"}:
        return
    published = json.loads(PUBLISHED_RUNS_PATH.read_text(encoding="utf-8"))
    source = published["runs"][config.SOURCE_RUN_ID]
    expected = {
        "max_sentences": source["max_sentences"],
        "token_count": source["token_count"],
        "token_stream_sha256_int64_le":
            source["token_stream_sha256_int64_le"],
    }
    mismatches = {
        key: (value, identity.get(key))
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatches:
        details = "; ".join(
            f"{key}: expected {want!r}, got {got!r}"
            for key, (want, got) in sorted(mismatches.items()))
        raise RuntimeError(
            f"derived run corpus does not match source "
            f"{config.SOURCE_RUN_ID!r} ({details})")


def main():
    manifest_path = config.RUN_DIR / "manifest.json"
    base = base_experiment()

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = existing.get("experiment")
        if (recorded == base
                and config.EXPERIMENT_TYPE == "substrate"
                and config.RUN_ID in config.LEGACY_RUN_CORPUS_IDENTITIES):
            verified_inputs()
            load_corpus(
                Tokenizer(),
                verify_manifest=False,
                return_identity=True,
            )
            print(f"validated legacy manifest and recovered corpus identity: "
                  f"{manifest_path}")
            return 0

    git_status = git_value("status", "--short")
    if git_status:
        print(
            "ERROR: refusing to create or resume a reproducible run from a "
            "dirty worktree. Commit or set aside the changes first.",
            file=sys.stderr,
        )
        print(git_status, file=sys.stderr)
        return 4

    inputs = verified_inputs()
    _, corpus_identity = load_corpus(
        Tokenizer(),
        verify_manifest=False,
        return_identity=True,
    )
    verify_trace_corpus_identity(corpus_identity)
    experiment = dict(base)
    experiment["experiment_type"] = config.EXPERIMENT_TYPE
    experiment["determinism"] = {
        "torch_deterministic_algorithms": True,
        "bitwise_cross_platform_claim": False,
    }
    experiment["corpus_identity"] = corpus_identity
    experiment["verified_inputs"] = inputs

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("experiment") != experiment:
            print(
                f"ERROR: {config.RUN_ID!r} already has a different config. "
                "Use a new STLM_RUN_ID.", file=sys.stderr)
            return 2
        current_runtime = runtime_identity()
        runtime_mismatches = {
            key: (existing.get(key), value)
            for key, value in current_runtime.items()
            if existing.get(key) != value
        }
        if runtime_mismatches:
            print(
                f"ERROR: {config.RUN_ID!r} was created under a different "
                "code or runtime environment. Use a new STLM_RUN_ID.",
                file=sys.stderr,
            )
            for key, (recorded_value, current_value) in sorted(
                    runtime_mismatches.items()):
                print(
                    f"  {key}: recorded={recorded_value!r}, "
                    f"current={current_value!r}",
                    file=sys.stderr,
                )
            return 3
        print(f"validated existing manifest: {manifest_path}")
        return 0

    runtime = runtime_identity()
    manifest = {
        "experiment": experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **runtime,
        "git_status": git_status,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"created run manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
