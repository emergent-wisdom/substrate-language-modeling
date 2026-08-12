"""
phase2_nn2.py -- train NN2 (CNN) to read NN1's pixel outputs as tokens.

NN1 is loaded from phase 1 and frozen. For each sampled context we project
only NN1's final predicted pixel image, then train NN2 with cross-entropy
to output the true next token. Text and pixel translators use the same
sampled context stream and fixed holdout positions.

    python phase2_nn2.py     # needs this run's completed pixel NN1 checkpoint
"""
import math

import torch
import torch.nn.functional as F

import config
from corpus import load_corpus
from models import ImageClassifier, ImagePredictor
from training_utils import (capture_torch_rng, install_interrupt_handler,
                            last_token_eval_batches,
                            last_token_eval_targets, record_metrics,
                            restore_torch_rng, seed_everything, stop_requested,
                            validate_checkpoint)
from vocab import Tokenizer


def main():
    config.require_experiment_type("substrate")
    config.require_writable_run()
    install_interrupt_handler()
    seed_everything(config.SEED)
    tok = Tokenizer()
    if not config.NN1_CKPT.exists():
        raise FileNotFoundError("run phase1_nn1.py first")

    ck = torch.load(config.NN1_CKPT, map_location=config.DEVICE)
    if not ck.get("run_complete", False):
        raise RuntimeError("pixel NN1 checkpoint is not complete")
    nn1 = ImagePredictor(
        n_tokens=ck["n_tokens"],
        img_size=ck["img_size"], img_ch=ck["img_ch"],
        context_len=ck["context_len"], txt_layers=ck["txt_layers"],
        txt_d_model=ck["txt_d_model"], txt_heads=ck["txt_heads"])
    nn1.load_state_dict(ck["state_dict"])
    nn1 = nn1.to(config.DEVICE)
    nn1.train(False)
    for p in nn1.parameters():
        p.requires_grad_(False)

    nn2 = ImageClassifier(
        config.IMG_SIZE, config.VOCAB_SIZE,
        channels=config.NN2_CHANNELS,
        fc_hidden=config.NN2_FC_HIDDEN).to(config.DEVICE)
    print(f"NN1 {nn1.n_params() / 1e6:.1f}M (frozen) | "
          f"NN2 {nn2.n_params() / 1e6:.2f}M (training) | {config.DEVICE}")
    opt = torch.optim.Adam(nn2.parameters(), lr=config.NN2_LR)

    stream = load_corpus(tok)
    T, B = config.NN1_CONTEXT, config.NN2_BATCH
    if len(stream) < T * 128:
        raise RuntimeError("corpus too small")
    n_eval = T * config.EVAL_CONTEXTS
    train_stream, eval_stream = stream[:-n_eval], stream[-n_eval:]
    eval_targets = last_token_eval_targets(
        eval_stream, T, config.NN2_EVAL_TARGETS)
    freq_counts = torch.bincount(eval_targets, minlength=len(tok))
    total_eval_targets = freq_counts.sum().item()
    freq_top = freq_counts.topk(10)
    print("eval frequency baseline "
          f"top1={freq_top.values[:1].sum().item() / total_eval_targets:.3f} "
          f"top5={freq_top.values[:5].sum().item() / total_eval_targets:.3f} "
          f"top10={freq_top.values[:10].sum().item() / total_eval_targets:.3f}")

    batch_rng = torch.Generator().manual_seed(config.SEED + 1_000)

    def get_batch(data_stream, bs):
        ix = torch.randint(
            0, len(data_stream) - T, (bs,), generator=batch_rng).tolist()
        inp = torch.stack(
            [data_stream[i:i + T] for i in ix]).to(config.DEVICE)
        nxt = torch.stack([data_stream[i + T] for i in ix]).to(config.DEVICE)
        return inp, nxt

    def step(inp, nxt):
        with torch.no_grad():
            imgs = nn1.forward_last(inp)                    # [B, 3, H, W]
        logits = nn2(imgs)
        loss = F.cross_entropy(logits, nxt)
        max_k = min(10, logits.shape[-1])
        pred = logits.topk(max_k, dim=-1).indices
        correct = pred.eq(nxt[:, None])
        metrics = {
            "top1": correct[:, :1].any(dim=1).float().mean().item(),
            "top5": correct[:, :min(5, max_k)].any(dim=1).float().mean().item(),
            "top10": correct[:, :max_k].any(dim=1).float().mean().item(),
        }
        return loss, metrics

    def evaluate():
        nn2.train(False)
        totals = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "top10": 0.0}
        n_targets = 0
        with torch.no_grad():
            for inp_e, nxt_e in last_token_eval_batches(
                    eval_stream, T, B, config.NN2_EVAL_TARGETS,
                    config.DEVICE):
                loss, metrics = step(inp_e, nxt_e)
                n = nxt_e.numel()
                totals["loss"] += loss.item() * n
                for name, value in metrics.items():
                    totals[name] += value * n
                n_targets += n
        return {name: value / n_targets for name, value in totals.items()}

    target_examples = config.NN2_TRAIN_TARGETS
    target_steps = math.ceil(target_examples / B)
    completed_steps = 0
    completed_examples = 0
    checkpoint_complete = False
    if config.NN2_CKPT.exists():
        prev = torch.load(config.NN2_CKPT, map_location=config.DEVICE)
        validate_checkpoint(prev, config.RUN_ID, config.SEED, target_steps)
        nn2.load_state_dict(prev["state_dict"])
        if "optimizer_state_dict" in prev:
            opt.load_state_dict(prev["optimizer_state_dict"])
        completed_steps = prev.get(
            "steps_trained", prev.get("global_steps", 0))
        completed_examples = prev.get("examples_trained", completed_steps * B)
        if prev.get("target_examples") != target_examples:
            raise RuntimeError("pixel NN2 checkpoint has a different target count")
        if "batch_rng_state" in prev:
            batch_rng.set_state(prev["batch_rng_state"].cpu())
        restore_torch_rng(prev.get("torch_rng_state"))
        checkpoint_complete = prev.get("run_complete", False)
        print(f"resumed from {config.NN2_CKPT.name}: "
              f"{completed_examples:,}/{target_examples:,} examples trained")
    else:
        print("no existing pixel NN2 checkpoint -- training from scratch")

    def save_checkpoint(done):
        torch.save({"state_dict": nn2.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "steps_trained": completed_steps,
                    "target_steps": target_steps,
                    "examples_trained": completed_examples,
                    "target_examples": target_examples,
                    "global_steps": completed_steps,
                    "run_complete": done,
                    "run_id": config.RUN_ID,
                    "seed": config.SEED,
                    "torch_rng_state": capture_torch_rng(),
                    "batch_rng_state": batch_rng.get_state(),
                    "nn1_steps_trained": ck["steps_trained"],
                    "img_size": config.IMG_SIZE,
                    "n_tokens": config.VOCAB_SIZE,
                    "channels": list(config.NN2_CHANNELS),
                    "fc_hidden": config.NN2_FC_HIDDEN}, config.NN2_CKPT)

    if checkpoint_complete:
        print("pixel NN2 checkpoint is already complete")
        return

    print(f"{target_examples - completed_examples:,} examples remaining "
          f"({target_steps - completed_steps} steps, checkpoint every "
          f"{config.NN2_SAVE_EVERY_STEPS} steps)")
    run_loss = run_top1 = run_top5 = run_top10 = 0.0
    window_targets = 0
    try:
        for step_idx in range(completed_steps, target_steps):
            if stop_requested():
                raise KeyboardInterrupt
            nn2.train()
            batch_size = min(B, target_examples - completed_examples)
            inp, nxt = get_batch(train_stream, batch_size)
            loss, metrics = step(inp, nxt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            completed_steps = step_idx + 1
            completed_examples += batch_size
            run_loss += loss.item() * batch_size
            run_top1 += metrics["top1"] * batch_size
            run_top5 += metrics["top5"] * batch_size
            run_top10 += metrics["top10"] * batch_size
            window_targets += batch_size

            if completed_steps % 50 == 0:
                print(f"step {completed_steps}/{target_steps}  "
                      f"train loss {run_loss / window_targets:.3f}  "
                      f"top1 {run_top1 / window_targets:.3f}  "
                      f"top5 {run_top5 / window_targets:.3f}  "
                      f"top10 {run_top10 / window_targets:.3f}")
                run_loss = run_top1 = run_top5 = run_top10 = 0.0
                window_targets = 0
            if completed_steps % config.NN2_SAVE_EVERY_STEPS == 0:
                save_checkpoint(False)
                print(f"autosaved step {completed_steps} -> "
                      f"{config.NN2_CKPT.name}")
            if stop_requested():
                raise KeyboardInterrupt
        if stop_requested():
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        save_checkpoint(False)
        print(f"interrupted at step {completed_steps}; checkpoint saved")
        raise SystemExit(130)

    ev = evaluate()
    print(f"eval loss {ev['loss']:.3f}  top1 {ev['top1']:.3f}  "
          f"top5 {ev['top5']:.3f}  top10 {ev['top10']:.3f}")
    record_metrics(config.RESULTS / "pixel_nn2.jsonl", {
        "eval_loss": ev["loss"],
        "eval_top1": ev["top1"],
        "eval_top5": ev["top5"],
        "eval_top10": ev["top10"],
        "evaluation_examples": total_eval_targets,
        "frequency_top1": freq_top.values[:1].sum().item() / total_eval_targets,
        "frequency_top5": freq_top.values[:5].sum().item() / total_eval_targets,
        "frequency_top10": freq_top.values[:10].sum().item() / total_eval_targets,
        "run_id": config.RUN_ID,
        "step": completed_steps,
        "training_examples": completed_examples,
    })
    save_checkpoint(True)
    print(f"completed {completed_examples:,} examples -> {config.NN2_CKPT}")


if __name__ == "__main__":
    main()
