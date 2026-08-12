"""Train contextual token readouts (TR-C) over frozen SP state sequences.

TR-C is one freshly initialized decoder-only causal Transformer. Every current
and prior substrate state enters that same model; there is no frozen
single-state readout, additive logit bypass, or raw-token input. The published
substrate predictor (SP) stays frozen.

Alignment is same-index at the seam. For token stream ``w`` and start ``s``::

    input[j]   = w[s + j]
    target[j]  = w[s + j + 1]
    state[j]   = SP(input)[j]
    logits[j]  = TR-C(state[0:j+1])

There is no second shift between ``state``, ``logits``, and ``target``.
Optimization uses all 128 positions in each of 100,000 sampled windows: 12.8
million supervised token positions and, at effective batch eight, 12,500
optimizer updates. Headline evaluation remains the published 1,024
deterministic final-position holdout targets.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from corpus import load_corpus
from models import (
    DescriptionContextualTokenReadout,
    DescriptionSubstratePredictor,
    PixelContextualTokenReadout,
    PixelSubstratePredictor,
)
from training_utils import (
    capture_torch_rng,
    fixed_eval_starts,
    install_interrupt_handler,
    last_token_eval_targets,
    record_metrics,
    restore_torch_rng,
    seed_everything,
    stop_requested,
    validate_checkpoint,
    verify_file_sha256,
)
from vocab import Tokenizer


PUBLISHED_RUNS_PATH = config.ROOT / "published" / "pilot_runs.json"


def aligned_windows(data, starts, context_len, device):
    """Build exact one-token-shifted input and target windows."""
    if not starts:
        raise ValueError("at least one window start is required")
    maximum_start = len(data) - context_len - 1
    invalid = [start for start in starts if not 0 <= start <= maximum_start]
    if invalid:
        raise ValueError(
            f"window starts must be in [0, {maximum_start}], got {invalid}")
    inputs = torch.stack([
        data[start:start + context_len]
        for start in starts
    ]).to(device)
    targets = torch.stack([
        data[start + 1:start + context_len + 1]
        for start in starts
    ]).to(device)
    if inputs.shape != targets.shape:
        raise RuntimeError("aligned input and target windows changed shape")
    if context_len > 1 and not torch.equal(inputs[:, 1:], targets[:, :-1]):
        raise RuntimeError("context windows are not shifted by exactly one")
    return inputs, targets


def all_position_loss(logits, targets, reduction="mean"):
    """Cross entropy over every same-index aligned sequence position."""
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("expected logits [B,T,V] and targets [B,T]")
    if logits.shape[:2] != targets.shape:
        raise ValueError("logit and target positions do not align")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction=reduction,
    )


def _published_checkpoint_hash(path):
    registry = json.loads(PUBLISHED_RUNS_PATH.read_text(encoding="utf-8"))
    try:
        return registry["runs"][config.SOURCE_RUN_ID][
            "checkpoint_sha256"][path.name]
    except KeyError as exc:
        raise RuntimeError(
            f"{path.name!r} is not registered for published source "
            f"{config.SOURCE_RUN_ID!r}") from exc


def _complete_source_checkpoint(path, label):
    expected = _published_checkpoint_hash(path)
    actual = verify_file_sha256(
        path, expected, f"{config.SOURCE_RUN_ID}/{path.name}")
    checkpoint = torch.load(path, map_location=config.DEVICE)
    if not checkpoint.get("run_complete", False):
        raise RuntimeError(f"source {label} checkpoint is not complete")
    if checkpoint.get("run_id") != config.SOURCE_RUN_ID:
        raise RuntimeError(
            f"source {label} belongs to {checkpoint.get('run_id')!r}, not "
            f"{config.SOURCE_RUN_ID!r}")
    return checkpoint, actual


def _freeze(module):
    module.train(False)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _load_pixel_components():
    sp_ck, sp_hash = _complete_source_checkpoint(
        config.SOURCE_PIXEL_SP_CKPT, "pixel SP")
    sp_metadata = {
        key: sp_ck[key]
        for key in (
            "n_tokens", "img_size", "img_ch", "context_len",
            "txt_layers", "txt_d_model", "txt_heads", "steps_trained",
        )
    }
    sp = PixelSubstratePredictor(
        n_tokens=sp_metadata["n_tokens"],
        img_size=sp_metadata["img_size"],
        img_ch=sp_metadata["img_ch"],
        context_len=sp_metadata["context_len"],
        txt_layers=sp_metadata["txt_layers"],
        txt_d_model=sp_metadata["txt_d_model"],
        txt_heads=sp_metadata["txt_heads"],
    ).to(config.DEVICE)
    sp.load_state_dict(sp_ck["state_dict"])
    _freeze(sp)

    seed_everything(config.CONTEXTUAL_READOUT_INIT_SEED)
    readout = PixelContextualTokenReadout(
        img_size=sp_metadata["img_size"],
        n_tokens=sp_metadata["n_tokens"],
        context_len=sp_metadata["context_len"],
        d_model=config.CONTEXTUAL_D_MODEL,
        n_heads=config.CONTEXTUAL_HEADS,
        n_layers=config.CONTEXTUAL_LAYERS,
        ff_dim=config.CONTEXTUAL_FF_DIM,
        pool_grid=config.CONTEXTUAL_PIXEL_POOL_GRID,
    ).to(config.DEVICE)
    # Do not retain the published checkpoint's unused Adam state in memory.
    return sp, readout, sp_metadata, {"sp": sp_hash}


def _load_description_components():
    sp_ck, sp_hash = _complete_source_checkpoint(
        config.SOURCE_DESCRIPTION_SP_CKPT, "description SP")
    sp_metadata = {
        key: sp_ck[key]
        for key in (
            "n_tokens", "desc_len", "embed_dim", "context_len",
            "txt_layers", "txt_d_model", "txt_heads", "steps_trained",
        )
    }
    sp = DescriptionSubstratePredictor(
        n_tokens=sp_metadata["n_tokens"],
        desc_len=sp_metadata["desc_len"],
        embed_dim=sp_metadata["embed_dim"],
        context_len=sp_metadata["context_len"],
        txt_layers=sp_metadata["txt_layers"],
        txt_d_model=sp_metadata["txt_d_model"],
        txt_heads=sp_metadata["txt_heads"],
    ).to(config.DEVICE)
    sp.load_state_dict(sp_ck["state_dict"])
    _freeze(sp)

    seed_everything(config.CONTEXTUAL_READOUT_INIT_SEED)
    readout = DescriptionContextualTokenReadout(
        embed_dim=sp_metadata["embed_dim"],
        desc_len=sp_metadata["desc_len"],
        n_tokens=sp_metadata["n_tokens"],
        context_len=sp_metadata["context_len"],
        d_model=config.CONTEXTUAL_D_MODEL,
        temporal_heads=config.CONTEXTUAL_HEADS,
        temporal_layers=config.CONTEXTUAL_LAYERS,
        temporal_ff_dim=config.CONTEXTUAL_FF_DIM,
        pool_bins=config.CONTEXTUAL_DESCRIPTION_POOL_BINS,
    ).to(config.DEVICE)
    # Do not retain the published checkpoint's unused Adam state in memory.
    return sp, readout, sp_metadata, {"sp": sp_hash}


def _medium_settings(medium):
    if medium == "pixel":
        return {
            "components": _load_pixel_components(),
            "logical_batch": config.CONTEXTUAL_PIXEL_BATCH,
            "microbatch": config.CONTEXTUAL_PIXEL_MICROBATCH,
            "checkpoint": config.CONTEXTUAL_PIXEL_CKPT,
            "metrics": config.RESULTS / "pixel_tr_c.jsonl",
            "predictions":
                config.RESULTS / "pixel_tr_c_eval_predictions.jsonl",
            "architecture": {
                "d_model": config.CONTEXTUAL_D_MODEL,
                "temporal_heads": config.CONTEXTUAL_HEADS,
                "temporal_layers": config.CONTEXTUAL_LAYERS,
                "temporal_ff_dim": config.CONTEXTUAL_FF_DIM,
                "pixel_pool_grid": config.CONTEXTUAL_PIXEL_POOL_GRID,
                "state_adapter":
                    "adaptive-average-pool, linear, GELU, LayerNorm",
            },
        }
    if medium == "description":
        return {
            "components": _load_description_components(),
            "logical_batch": config.CONTEXTUAL_DESCRIPTION_BATCH,
            "microbatch": config.CONTEXTUAL_DESCRIPTION_MICROBATCH,
            "checkpoint": config.CONTEXTUAL_DESCRIPTION_CKPT,
            "metrics": config.RESULTS / "description_tr_c.jsonl",
            "predictions":
                config.RESULTS / "description_tr_c_eval_predictions.jsonl",
            "architecture": {
                "d_model": config.CONTEXTUAL_D_MODEL,
                "temporal_heads": config.CONTEXTUAL_HEADS,
                "temporal_layers": config.CONTEXTUAL_LAYERS,
                "temporal_ff_dim": config.CONTEXTUAL_FF_DIM,
                "description_pool_bins":
                    config.CONTEXTUAL_DESCRIPTION_POOL_BINS,
                "state_adapter":
                    "ordered adaptive-average-pool, linear, GELU, "
                    "LayerNorm",
            },
        }
    raise ValueError(f"unknown medium: {medium!r}")


def _topk_counts(logits, targets):
    maximum_k = min(10, logits.shape[-1])
    predictions = logits.topk(maximum_k, dim=-1).indices
    correct = predictions.eq(targets[:, None])
    return {
        "top1": correct[:, :1].any(dim=1).sum().item(),
        "top5": correct[:, :min(5, maximum_k)].any(dim=1).sum().item(),
        "top10": correct.any(dim=1).sum().item(),
    }


def _evaluate(sp, readout, eval_stream, context_len, batch_size, tok):
    """Evaluate only the final aligned target in each fixed holdout window."""
    starts = fixed_eval_starts(
        eval_stream, context_len, config.READOUT_EVAL_TARGETS)
    totals = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0}
    records = []
    readout.train(False)
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset:offset + batch_size]
            inputs, targets = aligned_windows(
                eval_stream, batch_starts, context_len, config.DEVICE)
            trace = sp(inputs)
            logits = readout(trace)[:, -1]
            final_targets = targets[:, -1]
            totals["loss"] += F.cross_entropy(
                logits, final_targets, reduction="sum").item()
            for name, value in _topk_counts(logits, final_targets).items():
                totals[name] += value

            cpu_logits = logits.cpu()
            cpu_targets = final_targets.cpu().tolist()
            for local_index, target in enumerate(cpu_targets):
                one_logits = cpu_logits[local_index]
                log_probs = one_logits.log_softmax(dim=-1)
                top_ids = one_logits.topk(10).indices.tolist()
                records.append({
                    "eval_index": offset + local_index,
                    "holdout_start": batch_starts[local_index],
                    "target_id": target,
                    "target": tok.words[target],
                    "target_log_probability":
                        float(log_probs[target].item()),
                    "top10_ids": top_ids,
                    "top10_tokens": [tok.words[index] for index in top_ids],
                })
    count = len(starts)
    return ({
        "loss": totals["loss"] / count,
        "top1": totals["top1"] / count,
        "top5": totals["top5"] / count,
        "top10": totals["top10"] / count,
    }, records)


def _write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _existing_result(path, medium):
    path = Path(path)
    if not path.exists():
        return None
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in reversed(records):
        if (record.get("run_id") == config.RUN_ID
                and record.get("medium") == medium):
            return record
    return None


def train_contextual_readout(medium):
    config.require_experiment_type("contextual_readout")
    config.require_writable_run()
    install_interrupt_handler()
    seed_everything(config.SEED)
    tok = Tokenizer()
    settings = _medium_settings(medium)
    sp, readout, sp_ck, source_hashes = settings["components"]
    context_len = sp_ck["context_len"]
    logical_batch = settings["logical_batch"]
    microbatch = settings["microbatch"]
    checkpoint_path = settings["checkpoint"]
    trainable_parameters = list(readout.parameters())
    if not trainable_parameters or not all(
            parameter.requires_grad for parameter in trainable_parameters):
        raise RuntimeError("the complete integrated TR-C must be trainable")
    if any(parameter.requires_grad for parameter in sp.parameters()):
        raise RuntimeError("the published SP must remain frozen")

    training_windows = config.CONTEXTUAL_TRAIN_WINDOWS
    supervised_target_positions = training_windows * context_len
    target_steps = math.ceil(training_windows / logical_batch)
    hyperparameters = {
        "medium": medium,
        "objective": "all_aligned_positions",
        "context_length": context_len,
        "logical_batch": logical_batch,
        "microbatch": microbatch,
        "training_windows": training_windows,
        "supervised_target_positions": supervised_target_positions,
        "learning_rate": config.CONTEXTUAL_LR,
        "optimizer": "AdamW",
        "initialization": "fresh_deterministic",
        "initialization_seed": config.CONTEXTUAL_READOUT_INIT_SEED,
        "context_sampling_seed": config.CONTEXTUAL_CONTEXT_SAMPLING_SEED,
        "source_token_access": False,
        **settings["architecture"],
    }
    print(
        f"{medium} TR-C {readout.n_params() / 1e6:.2f}M trainable | "
        f"SP {sp.n_params() / 1e6:.2f}M frozen | context {context_len} | "
        f"logical batch {logical_batch}, microbatch {microbatch} | "
        f"{config.DEVICE}")
    print(
        f"budget: {training_windows:,} windows, "
        f"{supervised_target_positions:,} supervised positions, "
        f"{target_steps:,} optimizer updates")

    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=config.CONTEXTUAL_LR)
    sampling_rng = torch.Generator().manual_seed(
        config.CONTEXTUAL_CONTEXT_SAMPLING_SEED)
    completed_steps = 0
    completed_windows = 0
    checkpoint_complete = False
    if checkpoint_path.exists():
        previous = torch.load(checkpoint_path, map_location=config.DEVICE)
        validate_checkpoint(
            previous, config.RUN_ID, config.SEED, target_steps)
        if previous.get("source_run_id") != config.SOURCE_RUN_ID:
            raise RuntimeError("TR-C checkpoint uses a different source run")
        if previous.get("source_checkpoint_sha256") != source_hashes:
            raise RuntimeError("TR-C source checkpoint hashes changed")
        if previous.get("hyperparameters") != hyperparameters:
            raise RuntimeError("TR-C checkpoint hyperparameters changed")
        readout.load_state_dict(previous["state_dict"])
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        completed_steps = previous["steps_trained"]
        completed_windows = previous["completed_windows"]
        expected_positions = completed_windows * context_len
        if previous.get("completed_target_positions") != expected_positions:
            raise RuntimeError(
                "TR-C checkpoint window and target-position counts disagree")
        sampling_rng.set_state(previous["sampling_rng_state"].cpu())
        restore_torch_rng(previous.get("torch_rng_state"))
        checkpoint_complete = previous.get("run_complete", False)
        print(
            f"resumed {medium} TR-C: {completed_windows:,}/"
            f"{training_windows:,} windows "
            f"({expected_positions:,} supervised positions)")

    existing = _existing_result(settings["metrics"], medium)
    if (checkpoint_complete and existing is not None
            and settings["predictions"].exists()):
        print(f"{medium} TR-C already complete; outputs are intact")
        return existing

    stream = load_corpus(tok)
    n_eval = context_len * config.EVAL_CONTEXTS
    train_stream, eval_stream = stream[:-n_eval], stream[-n_eval:]
    if len(train_stream) <= context_len:
        raise RuntimeError("training corpus is too small for one context")

    def save_checkpoint(done, evaluation=None):
        completed_target_positions = completed_windows * context_len
        torch.save({
            "state_dict": readout.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "steps_trained": completed_steps,
            "target_steps": target_steps,
            "optimizer_updates": completed_steps,
            "training_windows": training_windows,
            "supervised_target_positions": supervised_target_positions,
            "completed_windows": completed_windows,
            "completed_target_positions": completed_target_positions,
            "objective": "all_aligned_positions",
            "context_length": context_len,
            "run_complete": done,
            "run_id": config.RUN_ID,
            "seed": config.SEED,
            "source_run_id": config.SOURCE_RUN_ID,
            "source_checkpoint_sha256": source_hashes,
            "source_sp_steps_trained": sp_ck["steps_trained"],
            "torch_rng_state": capture_torch_rng(),
            "sampling_rng_state": sampling_rng.get_state(),
            "hyperparameters": hyperparameters,
            "evaluation": evaluation,
        }, checkpoint_path)

    if not checkpoint_complete:
        rolling_loss = rolling_correct = rolling_positions = 0
        progress = tqdm(
            total=target_steps,
            initial=completed_steps,
            desc=f"{medium} TR-C",
            mininterval=2.0,
        )
        try:
            for step_index in range(completed_steps, target_steps):
                if stop_requested():
                    raise KeyboardInterrupt
                current_batch = min(
                    logical_batch, training_windows - completed_windows)
                starts = torch.randint(
                    0,
                    len(train_stream) - context_len,
                    (current_batch,),
                    generator=sampling_rng,
                ).tolist()
                readout.train()
                optimizer.zero_grad()
                step_loss = step_correct = 0
                step_positions = current_batch * context_len
                for offset in range(0, current_batch, microbatch):
                    micro_starts = starts[offset:offset + microbatch]
                    inputs, targets = aligned_windows(
                        train_stream, micro_starts, context_len, config.DEVICE)
                    with torch.no_grad():
                        trace = sp(inputs)
                    logits = readout(trace)
                    loss_sum = all_position_loss(
                        logits, targets, reduction="sum")
                    (loss_sum / step_positions).backward()
                    step_loss += loss_sum.item()
                    step_correct += logits.argmax(dim=-1).eq(
                        targets).sum().item()
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=float("inf"),
                    error_if_nonfinite=True,
                )
                optimizer.step()

                completed_steps = step_index + 1
                completed_windows += current_batch
                rolling_loss += step_loss
                rolling_correct += step_correct
                rolling_positions += step_positions
                progress.update(1)
                if completed_steps % 50 == 0:
                    progress.set_postfix(
                        loss=f"{rolling_loss / rolling_positions:.3f}",
                        top1=f"{rolling_correct / rolling_positions:.3f}",
                    )
                    rolling_loss = rolling_correct = rolling_positions = 0
                if completed_steps % config.CONTEXTUAL_SAVE_EVERY_STEPS == 0:
                    save_checkpoint(False)
                    progress.write(
                        f"autosaved step {completed_steps} -> "
                        f"{checkpoint_path.name}")
                if stop_requested():
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            save_checkpoint(False)
            progress.write(
                f"interrupted at step {completed_steps}; checkpoint saved")
            raise SystemExit(130)
        finally:
            progress.close()

    evaluation, predictions = _evaluate(
        sp=sp,
        readout=readout,
        eval_stream=eval_stream,
        context_len=context_len,
        batch_size=microbatch,
        tok=tok,
    )
    _write_jsonl(settings["predictions"], predictions)
    final_targets = last_token_eval_targets(
        eval_stream, context_len, config.READOUT_EVAL_TARGETS)
    counts = torch.bincount(final_targets, minlength=len(tok)).topk(10).values
    denominator = final_targets.numel()
    result = {
        "run_id": config.RUN_ID,
        "source_run_id": config.SOURCE_RUN_ID,
        "medium": medium,
        "readout": "TR-C",
        "objective": "all_aligned_positions",
        "context_length": context_len,
        "eval_loss": evaluation["loss"],
        "eval_perplexity": math.exp(evaluation["loss"]),
        "eval_top1": evaluation["top1"],
        "eval_top5": evaluation["top5"],
        "eval_top10": evaluation["top10"],
        "evaluation_examples": len(predictions),
        "frequency_top1": counts[:1].sum().item() / denominator,
        "frequency_top5": counts[:5].sum().item() / denominator,
        "frequency_top10": counts[:10].sum().item() / denominator,
        "step": completed_steps,
        "optimizer_updates": completed_steps,
        "training_windows": completed_windows,
        "supervised_target_positions": completed_windows * context_len,
        "params": readout.n_params(),
        "source_checkpoint_sha256": source_hashes,
        "hyperparameters": hyperparameters,
    }
    if not checkpoint_complete:
        save_checkpoint(True, evaluation=result)
    if existing is None:
        record_metrics(settings["metrics"], result)
    elif existing != result:
        raise RuntimeError("existing TR-C metrics differ from this evaluation")
    print(
        f"eval loss {evaluation['loss']:.3f}  "
        f"top1 {evaluation['top1']:.3f}  top5 {evaluation['top5']:.3f}  "
        f"top10 {evaluation['top10']:.3f}")
    print(f"completed -> {checkpoint_path}")
    return result
