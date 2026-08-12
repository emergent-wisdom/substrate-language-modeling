"""
phase2_nn2_text.py -- train NN2 to read NN1's text-embedding substrate.

NN1 is frozen. For each sampled context we project its final predicted
[L, TEXT_DIM] embedding sequence; NN2 reads that substrate and predicts
the true next word via cross-entropy.

NN2 receives no story tokens and no canonical attention mask. The only
channel between NN1 and NN2 is the substrate sequence NN1 emits. PAD
positions are represented by vector content, not by a side-channel mask.

Resumable: the run trains on NN2_TRAIN_TARGETS sampled next-token contexts
and saves model, optimizer, completed-example, and RNG state.

    python phase2_nn2_text.py
"""
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from corpus import load_corpus
from models import DescriptionEmbeddingClassifier, DescriptionEmbeddingPredictor
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
    if not config.NN1_TEXT_CKPT.exists():
        raise FileNotFoundError("run phase1_nn1_text.py first")

    ck = torch.load(config.NN1_TEXT_CKPT, map_location=config.DEVICE)
    if not ck.get("run_complete", False):
        raise RuntimeError("text NN1 checkpoint is not complete")
    nn1 = DescriptionEmbeddingPredictor(
        n_tokens=ck["n_tokens"],
        desc_len=ck["desc_len"],
        embed_dim=ck["embed_dim"],
        context_len=ck["context_len"],
        txt_layers=ck["txt_layers"],
        txt_d_model=ck["txt_d_model"],
        txt_heads=ck["txt_heads"])
    nn1.load_state_dict(ck["state_dict"])
    nn1 = nn1.to(config.DEVICE)
    nn1.train(False)
    for p in nn1.parameters():
        p.requires_grad_(False)

    nn2 = DescriptionEmbeddingClassifier(
        embed_dim=ck["embed_dim"],
        n_tokens=len(tok),
        desc_len=ck["desc_len"],
        d_model=config.NN2_TEXT_D_MODEL,
        n_heads=config.NN2_TEXT_HEADS,
        n_layers=config.NN2_TEXT_LAYERS,
        ff_dim=config.NN2_TEXT_FF_DIM).to(config.DEVICE)
    print(f"NN1 (text-embedding) {nn1.n_params() / 1e6:.2f}M frozen | "
          f"NN2 {nn2.n_params() / 1e6:.2f}M training | {config.DEVICE}")
    opt = torch.optim.AdamW(nn2.parameters(), lr=config.NN2_LR)

    stream = load_corpus(tok)
    T, B = config.NN1_CONTEXT, config.NN2_TEXT_BATCH
    if len(stream) < T * 128:
        raise RuntimeError("corpus too small")
    n_eval = T * config.EVAL_CONTEXTS
    train_stream, eval_stream = stream[:-n_eval], stream[-n_eval:]
    eval_targets = last_token_eval_targets(
        eval_stream, T, config.NN2_EVAL_TARGETS)
    freq_counts = torch.bincount(eval_targets, minlength=len(tok))
    total_eval_targets = freq_counts.sum().item()
    freq_top = freq_counts.topk(10)
    freq_top1 = freq_top.values[:1].sum().item() / total_eval_targets
    freq_top5 = freq_top.values[:5].sum().item() / total_eval_targets
    freq_top10 = freq_top.values[:10].sum().item() / total_eval_targets
    top_word = tok.words[freq_top.indices[0].item()]
    print("eval frequency baseline "
          f"top1={freq_top1:.3f} ({top_word!r}) "
          f"top5={freq_top5:.3f} top10={freq_top10:.3f}")

    batch_rng = torch.Generator().manual_seed(config.SEED + 1_000)

    def get_batch(data_stream, bs):
        ix = torch.randint(
            0, len(data_stream) - T, (bs,), generator=batch_rng).tolist()
        inp = torch.stack(
            [data_stream[i:i + T] for i in ix]).to(config.DEVICE)
        nxt = torch.stack([data_stream[i + T] for i in ix]).to(config.DEVICE)
        return inp, nxt

    def topk_metrics(logits, labels):
        max_k = min(10, logits.shape[-1])
        pred = logits.topk(max_k, dim=-1).indices
        correct = pred.eq(labels[:, None])
        return {
            "top1": correct[:, :1].any(dim=1).float().mean().item(),
            "top5": correct[:, :min(5, max_k)].any(dim=1).float().mean().item(),
            "top10": correct[:, :max_k].any(dim=1).float().mean().item(),
        }

    def forward_batch(inp, nxt):
        with torch.no_grad():
            pred_seq = nn1.forward_last(inp)                       # [B, L, D]
        logits = nn2(pred_seq)
        loss = F.cross_entropy(logits, nxt)
        metrics = topk_metrics(logits, nxt)
        return loss, metrics

    def evaluate():
        nn2.train(False)
        totals = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "top10": 0.0}
        n_targets = 0
        with torch.no_grad():
            for inp_e, nxt_e in last_token_eval_batches(
                    eval_stream, T, B, config.NN2_EVAL_TARGETS,
                    config.DEVICE):
                loss, metrics = forward_batch(inp_e, nxt_e)
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
    if config.NN2_TEXT_CKPT.exists():
        prev = torch.load(config.NN2_TEXT_CKPT, map_location=config.DEVICE)
        validate_checkpoint(prev, config.RUN_ID, config.SEED, target_steps)
        nn2.load_state_dict(prev["state_dict"])
        if "optimizer_state_dict" in prev:
            opt.load_state_dict(prev["optimizer_state_dict"])
        completed_steps = prev.get(
            "steps_trained", prev.get("global_steps", 0))
        completed_examples = prev.get("examples_trained", completed_steps * B)
        if prev.get("target_examples") != target_examples:
            raise RuntimeError("text NN2 checkpoint has a different target count")
        if "batch_rng_state" in prev:
            batch_rng.set_state(prev["batch_rng_state"].cpu())
        restore_torch_rng(prev.get("torch_rng_state"))
        checkpoint_complete = prev.get("run_complete", False)
        print(f"resumed NN2 from {config.NN2_TEXT_CKPT.name}: "
              f"{completed_examples:,}/{target_examples:,} examples trained")
    else:
        print("no existing text NN2 checkpoint -- training from scratch")

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
                    "embed_dim": ck["embed_dim"],
                    "desc_len": ck["desc_len"],
                    "n_tokens": len(tok),
                    "d_model": config.NN2_TEXT_D_MODEL,
                    "n_heads": config.NN2_TEXT_HEADS,
                    "n_layers": config.NN2_TEXT_LAYERS,
                    "ff_dim": config.NN2_TEXT_FF_DIM}, config.NN2_TEXT_CKPT)

    if checkpoint_complete:
        print("text NN2 checkpoint is already complete")
        return

    print(f"{target_examples - completed_examples:,} examples remaining "
          f"({target_steps - completed_steps} steps, checkpoint every "
          f"{config.NN2_TEXT_SAVE_EVERY_STEPS} steps)")
    run_loss = run_top1 = run_top5 = run_top10 = 0.0
    window_targets = 0
    pbar = tqdm(total=target_steps, initial=completed_steps,
                desc="text NN2", mininterval=2.0)
    try:
        for step_idx in range(completed_steps, target_steps):
            if stop_requested():
                raise KeyboardInterrupt
            nn2.train()
            batch_size = min(B, target_examples - completed_examples)
            inp, nxt = get_batch(train_stream, batch_size)
            loss, metrics = forward_batch(inp, nxt)
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
            pbar.update(1)
            if completed_steps % 50 == 0:
                pbar.set_postfix(
                    loss=f"{run_loss/window_targets:.3f}",
                    top1=f"{run_top1/window_targets:.3f}",
                    top5=f"{run_top5/window_targets:.3f}",
                    top10=f"{run_top10/window_targets:.3f}")
                run_loss = run_top1 = run_top5 = run_top10 = 0.0
                window_targets = 0
            if completed_steps % config.NN2_TEXT_SAVE_EVERY_STEPS == 0:
                save_checkpoint(False)
                pbar.write(
                    f"autosaved step {completed_steps} -> "
                    f"{config.NN2_TEXT_CKPT.name}")
            if stop_requested():
                raise KeyboardInterrupt
        if stop_requested():
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        save_checkpoint(False)
        pbar.write(
            f"interrupted at step {completed_steps}; checkpoint saved")
        raise SystemExit(130)
    finally:
        pbar.close()

    ev = evaluate()
    print(f"eval loss {ev['loss']:.3f}  top1 {ev['top1']:.3f}  "
          f"top5 {ev['top5']:.3f}  top10 {ev['top10']:.3f}")
    record_metrics(config.RESULTS / "text_nn2.jsonl", {
        "eval_loss": ev["loss"],
        "eval_top1": ev["top1"],
        "eval_top5": ev["top5"],
        "eval_top10": ev["top10"],
        "evaluation_examples": total_eval_targets,
        "frequency_top1": freq_top1,
        "frequency_top5": freq_top5,
        "frequency_top10": freq_top10,
        "run_id": config.RUN_ID,
        "step": completed_steps,
        "training_examples": completed_examples,
    })
    save_checkpoint(True)
    print(f"completed {completed_examples:,} examples -> {config.NN2_TEXT_CKPT}")


if __name__ == "__main__":
    main()
