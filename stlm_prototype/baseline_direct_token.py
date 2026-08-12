"""
baseline_direct_token.py -- backbone-matched direct-token LM baseline.

The same causal transformer body as the pixel NN1 (layers, width, heads,
context), the same corpus slice, seed, batch size, optimizer, learning
rate, and step count -- but with an ordinary token-softmax head instead
of a substrate head plus a separate translator. There is no seam and no
NN2.

Matched: backbone architecture and initialization, token exposure,
optimizer updates, and evaluation contexts. NOT matched: total capacity
(11.48M parameters here against 13.45M for the description system and
18.31M for the pixel system) or total training compute (the substrate
systems additionally fit a translator over 100,000 sampled contexts).
The asymmetry is conservative if this baseline wins -- it would win with
less of both -- but a loss is correspondingly not symmetric evidence.
Report it as backbone-matched, never as fully matched.

Evaluated at the final position of the same fixed holdout contexts the
substrate translators use, so its top-k numbers are directly comparable
to the substrate rows. Also reports next-token NLL/perplexity and the
oracle test-marginal frequency baseline; the latter should reproduce the
substrate runs' frequency figures exactly, which doubles as a check that
this harness evaluates identically.

Use a dedicated run ID so the baseline receives its own immutable manifest:

    STLM_EXPERIMENT_TYPE=direct_token STLM_RUN_ID=baseline5x_seed42 \
        STLM_MAX_SENTENCES=250000 \
        python write_run_manifest.py
    STLM_EXPERIMENT_TYPE=direct_token STLM_RUN_ID=baseline5x_seed42 \
        STLM_MAX_SENTENCES=250000 \
        python baseline_direct_token.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from phase1_nn1 import load_corpus
from training_utils import (capture_torch_rng, install_interrupt_handler,
                            last_token_eval_batches, last_token_eval_targets,
                            record_metrics, restore_torch_rng,
                            seed_everything, stop_requested,
                            validate_checkpoint)
from vocab import Tokenizer

CKPT = config.CHECKPOINTS / "baseline_direct_token.pt"
METRICS = config.RESULTS / "baseline_direct_token.jsonl"


class TokenPredictor(nn.Module):
    """Direct-token LM. Body identical to ImagePredictor; head -> vocabulary."""

    def __init__(self, n_tokens, context_len,
                 txt_layers=6, txt_d_model=384, txt_heads=6):
        super().__init__()
        self.context_len = context_len
        self.tok_emb = nn.Embedding(n_tokens, txt_d_model)
        self.pos_emb = nn.Embedding(context_len, txt_d_model)
        block = nn.TransformerEncoderLayer(
            txt_d_model, txt_heads, dim_feedforward=4 * txt_d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(
            block, txt_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(txt_d_model)
        self.head = nn.Linear(txt_d_model, n_tokens)

    def _encode(self, ids):
        T = ids.shape[1]
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None]
        mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=ids.device)
        x = self.blocks(x, mask=mask)
        return self.norm(x)

    def forward(self, ids):
        """ids: [B, T] -> [B, T, n_tokens] logits."""
        return self.head(self._encode(ids))

    def forward_last(self, ids):
        """Logits for the final context position only."""
        return self.head(self._encode(ids)[:, -1])

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def topk_metrics(logits, target):
    max_k = min(10, logits.shape[-1])
    pred = logits.topk(max_k, dim=-1).indices
    correct = pred.eq(target[:, None])
    return {
        "top1": correct[:, :1].any(dim=1).float().mean().item(),
        "top5": correct[:, :min(5, max_k)].any(dim=1).float().mean().item(),
        "top10": correct[:, :max_k].any(dim=1).float().mean().item(),
    }


def frequency_baseline(targets, n_tokens):
    counts = torch.bincount(targets, minlength=n_tokens).float()
    order = counts.argsort(descending=True)
    total = targets.numel()
    out = {}
    for k in (1, 5, 10):
        hits = counts[order[:k]].sum().item()
        out[f"frequency_top{k}"] = hits / total
    return out


def main():
    if config.EXPERIMENT_TYPE != "direct_token":
        raise RuntimeError(
            "set STLM_EXPERIMENT_TYPE=direct_token for this baseline")
    config.require_writable_run()
    install_interrupt_handler()
    seed_everything(config.SEED)
    tok = Tokenizer()

    model = TokenPredictor(
        n_tokens=len(tok), context_len=config.NN1_CONTEXT,
        txt_layers=config.NN1_TXT_LAYERS, txt_d_model=config.NN1_TXT_D_MODEL,
        txt_heads=config.NN1_TXT_HEADS).to(config.DEVICE)
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"direct-token LM {model.n_params() / 1e6:.2f}M params "
          f"({head_params / 1e6:.2f}M in the token head) on {config.DEVICE}")
    opt = torch.optim.AdamW(model.parameters(), lr=config.NN1_LR)

    stream = load_corpus(tok)
    T, B = config.NN1_CONTEXT, config.NN1_BATCH
    n_eval = T * config.EVAL_CONTEXTS
    train_stream, eval_stream = stream[:-n_eval], stream[-n_eval:]
    train_batch_rng = torch.Generator().manual_seed(config.SEED + 1_000)

    def get_batch(data_stream, bs, generator):
        ix = torch.randint(
            0, len(data_stream) - T - 1, (bs,), generator=generator).tolist()
        inp = torch.stack(
            [data_stream[i:i + T] for i in ix]).to(config.DEVICE)
        nxt = torch.stack(
            [data_stream[i + 1:i + 1 + T] for i in ix]).to(config.DEVICE)
        return inp, nxt

    steps = max(1, len(train_stream) // (B * T))
    target_steps = config.NN1_EPOCHS * steps
    completed_steps = 0
    already_complete = False
    if CKPT.exists():
        prev = torch.load(CKPT, map_location=config.DEVICE)
        validate_checkpoint(prev, config.RUN_ID, config.SEED, target_steps)
        already_complete = prev.get("run_complete", False)
        model.load_state_dict(prev["state_dict"])
        if "optimizer_state_dict" in prev:
            opt.load_state_dict(prev["optimizer_state_dict"])
        completed_steps = prev.get("steps_trained", 0)
        if "train_batch_rng_state" in prev:
            train_batch_rng.set_state(prev["train_batch_rng_state"].cpu())
        restore_torch_rng(prev.get("torch_rng_state"))
        print(f"resumed: {completed_steps}/{target_steps} steps trained")
    else:
        print("no existing baseline checkpoint -- training from scratch")

    def save_checkpoint(done):
        torch.save({"state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "steps_trained": completed_steps,
                    "target_steps": target_steps,
                    "run_complete": done,
                    "run_id": config.RUN_ID,
                    "seed": config.SEED,
                    "torch_rng_state": capture_torch_rng(),
                    "train_batch_rng_state": train_batch_rng.get_state(),
                    "n_tokens": len(tok),
                    "context_len": config.NN1_CONTEXT,
                    "txt_layers": config.NN1_TXT_LAYERS,
                    "txt_d_model": config.NN1_TXT_D_MODEL,
                    "txt_heads": config.NN1_TXT_HEADS}, CKPT)

    def evaluate():
        """Final-position metrics on the same fixed contexts NN2 is scored on."""
        model.train(False)
        totals = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "top10": 0.0}
        n = 0
        with torch.no_grad():
            for inp_e, nxt_e in last_token_eval_batches(
                    eval_stream, T, config.NN2_BATCH,
                    config.NN2_EVAL_TARGETS, config.DEVICE):
                logits = model.forward_last(inp_e)
                loss = F.cross_entropy(logits, nxt_e)
                m = topk_metrics(logits, nxt_e)
                bs = nxt_e.numel()
                totals["loss"] += loss.item() * bs
                for name, value in m.items():
                    totals[name] += value * bs
                n += bs
        model.train(True)
        return {name: value / n for name, value in totals.items()}, n

    print(f"{target_steps - completed_steps} steps remaining "
          f"({steps} steps/epoch, batch {B}, context {T})")
    run_loss, window = 0.0, 0
    try:
        for step_idx in range(completed_steps, target_steps):
            if stop_requested():
                raise KeyboardInterrupt
            model.train()
            inp, nxt = get_batch(train_stream, B, train_batch_rng)
            logits = model(inp)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), nxt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            completed_steps = step_idx + 1
            run_loss += loss.item()
            window += 1

            if completed_steps % 50 == 0:
                print(f"step {completed_steps}/{target_steps}  "
                      f"train CE {run_loss / window:.4f}")
                run_loss, window = 0.0, 0
            if completed_steps % config.NN1_SAVE_EVERY_STEPS == 0:
                save_checkpoint(False)
                print(f"autosaved step {completed_steps}")
            if completed_steps % steps == 0:
                ev, n_targets = evaluate()
                print(f"epoch {completed_steps // steps}  "
                      f"eval CE {ev['loss']:.4f}  ppl {math.exp(ev['loss']):.1f}  "
                      f"top1 {100 * ev['top1']:.1f}%  "
                      f"top5 {100 * ev['top5']:.1f}%  "
                      f"top10 {100 * ev['top10']:.1f}%")
                save_checkpoint(completed_steps >= target_steps)
    except KeyboardInterrupt:
        save_checkpoint(False)
        print(f"interrupted at step {completed_steps}; checkpoint saved")
        raise SystemExit(130)

    ev, n_targets = evaluate()
    freq = frequency_baseline(
        last_token_eval_targets(eval_stream, T, config.NN2_EVAL_TARGETS),
        len(tok))
    record = {"run_id": config.RUN_ID, "step": completed_steps,
              "params": model.n_params(), "head_params": head_params,
              "eval_loss": ev["loss"], "eval_perplexity": math.exp(ev["loss"]),
              "eval_top1": ev["top1"], "eval_top5": ev["top5"],
              "eval_top10": ev["top10"],
              "evaluation_examples": n_targets, **freq}
    if already_complete:
        print("(checkpoint was already complete; not appending a duplicate "
              "metrics record)")
    else:
        record_metrics(METRICS, record)
    save_checkpoint(True)
    print(f"\ndirect-token baseline, {completed_steps} steps, "
          f"{model.n_params() / 1e6:.2f}M params")
    print(f"  eval CE {ev['loss']:.4f}  perplexity {math.exp(ev['loss']):.1f}")
    print(f"  top1 {100 * ev['top1']:.1f}%  top5 {100 * ev['top5']:.1f}%"
          f"  top10 {100 * ev['top10']:.1f}%")
    print(f"  oracle frequency floor: top1 {100 * freq['frequency_top1']:.1f}%"
          f"  top5 {100 * freq['frequency_top5']:.1f}%"
          f"  top10 {100 * freq['frequency_top10']:.1f}%")
    print(f"  metrics -> {METRICS}")


if __name__ == "__main__":
    main()
