"""
phase1_nn1_text.py -- train NN1 for the text-embedding substrate.

NN1 sees context token IDs only. At each position, its target is the
[TEXT_DESC_LEN, TEXT_DIM] per-token BERT hidden-state sequence for the
next word's description. NN1 emits unit-norm vectors at the seam; loss
is cosine distance per position, averaged over ALL positions (real BERT
states at valid slots, the fixed PAD vector at padded slots). We also
report a *content-only* mean cosine -- averaged over `mask == 1`
positions only -- so you can tell whether NN1 is learning real
description content vs. just learning to emit PAD after the description
ends.

Resumable: NN1_TEXT_EPOCHS is a total target. The run saves model,
optimizer, completed-step, and RNG state during each epoch, then continues
from that exact target when relaunched with the same STLM_RUN_ID.

    python phase1_nn1_text.py     # trains to the NN1_TEXT_EPOCHS total target
"""
import numpy as np
import torch
from tqdm import tqdm

import config
from corpus import load_corpus
from models import DescriptionEmbeddingPredictor
from training_utils import (capture_torch_rng, install_interrupt_handler,
                            record_metrics, restore_torch_rng,
                            seed_everything, stop_requested,
                            validate_checkpoint, verify_file_sha256)
from vocab import Tokenizer


def main():
    config.require_experiment_type("substrate")
    config.require_writable_run()
    install_interrupt_handler()
    seed_everything(config.SEED)
    tok = Tokenizer()
    if not config.TEXT_CANONICALS_FILE.exists():
        raise FileNotFoundError("run phase0_text_canonicals.py first")
    verify_file_sha256(
        config.TEXT_CANONICALS_FILE,
        config.TEXT_CANONICALS_SHA256,
        "text canonicals",
    )

    ck_npz = np.load(config.TEXT_CANONICALS_FILE)
    canonicals = torch.from_numpy(ck_npz["canonicals"]).to(config.DEVICE)
    canon_mask = torch.from_numpy(ck_npz["mask"]).to(config.DEVICE)
    # [VOCAB, L, D] -- every position unit-norm (valid BERT states at
    # real positions, fixed PAD vector at padded positions).
    # [VOCAB, L] mask kept for diagnostics only.
    print(f"text canonicals: {tuple(canonicals.shape)} "
          f"(valid avg length {canon_mask.sum(dim=1).float().mean().item():.1f}/{canon_mask.shape[1]})")

    nn1 = DescriptionEmbeddingPredictor(
        n_tokens=len(tok),
        desc_len=config.TEXT_DESC_LEN,
        embed_dim=config.TEXT_DIM,
        context_len=config.NN1_CONTEXT,
        txt_layers=config.NN1_TXT_LAYERS,
        txt_d_model=config.NN1_TXT_D_MODEL,
        txt_heads=config.NN1_TXT_HEADS).to(config.DEVICE)
    print(f"NN1 (text-embedding) {nn1.n_params() / 1e6:.2f}M params on {config.DEVICE}")
    opt = torch.optim.AdamW(nn1.parameters(), lr=config.NN1_LR)

    stream = load_corpus(tok)
    T, B = config.NN1_CONTEXT, config.NN1_TEXT_BATCH
    if len(stream) < T * 128:
        raise RuntimeError("corpus too small; raise MAX_SENTENCES")
    n_eval = T * config.EVAL_CONTEXTS
    train_stream, eval_stream = stream[:-n_eval], stream[-n_eval:]
    train_batch_rng = torch.Generator().manual_seed(config.SEED + 1_000)
    eval_batch_rng = torch.Generator().manual_seed(config.SEED + 2_000)

    def get_batch(data_stream, bs, generator):
        ix = torch.randint(
            0, len(data_stream) - T - 1, (bs,),
            generator=generator).tolist()
        inp = torch.stack(
            [data_stream[i:i + T] for i in ix]).to(config.DEVICE)
        nxt = torch.stack(
            [data_stream[i + 1:i + 1 + T] for i in ix]).to(config.DEVICE)
        return inp, nxt

    def step(inp, nxt):
        pred = nn1(inp)                       # [B, T, L, D], unit-norm
        tgt = canonicals[nxt]                 # [B, T, L, D], unit-norm at ALL positions
        m = canon_mask[nxt].float()           # [B, T, L]: 1 at real content slots
        # All positions are supervised: real BERT states at valid slots,
        # the fixed PAD vector at padded slots.
        cos = (pred * tgt).sum(dim=-1)        # [B, T, L]
        loss = (1.0 - cos).mean()
        with torch.no_grad():
            all_cos = cos.mean()
            n_content = m.sum().clamp(min=1.0)
            content_cos = (cos * m).sum() / n_content
        return loss, all_cos.item(), content_cos.item()

    steps = max(1, len(train_stream) // (B * T))
    completed_steps = 0
    target_steps = config.NN1_TEXT_EPOCHS * steps
    if config.NN1_TEXT_CKPT.exists():
        prev = torch.load(config.NN1_TEXT_CKPT, map_location=config.DEVICE)
        validate_checkpoint(prev, config.RUN_ID, config.SEED, target_steps)
        nn1.load_state_dict(prev["state_dict"])
        if "optimizer_state_dict" in prev:
            opt.load_state_dict(prev["optimizer_state_dict"])
        completed_steps = prev.get(
            "steps_trained", prev.get("global_steps", 0))
        if "train_batch_rng_state" in prev:
            train_batch_rng.set_state(prev["train_batch_rng_state"].cpu())
        if "eval_batch_rng_state" in prev:
            eval_batch_rng.set_state(prev["eval_batch_rng_state"].cpu())
        restore_torch_rng(prev.get("torch_rng_state"))
        print(f"resumed from {config.NN1_TEXT_CKPT.name}: "
              f"{completed_steps}/{target_steps} steps trained")
    else:
        print("no existing text NN1 checkpoint -- training from scratch")

    def save_checkpoint(done):
        torch.save({"state_dict": nn1.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epochs_trained": completed_steps // steps,
                    "steps_trained": completed_steps,
                    "target_steps": target_steps,
                    "global_steps": completed_steps,
                    "run_complete": done,
                    "run_id": config.RUN_ID,
                    "seed": config.SEED,
                    "torch_rng_state": capture_torch_rng(),
                    "train_batch_rng_state": train_batch_rng.get_state(),
                    "eval_batch_rng_state": eval_batch_rng.get_state(),
                    "n_tokens": len(tok),
                    "desc_len": config.TEXT_DESC_LEN,
                    "embed_dim": config.TEXT_DIM,
                    "context_len": config.NN1_CONTEXT,
                    "txt_layers": config.NN1_TXT_LAYERS,
                    "txt_d_model": config.NN1_TXT_D_MODEL,
                    "txt_heads": config.NN1_TXT_HEADS}, config.NN1_TEXT_CKPT)

    print(f"{target_steps - completed_steps} steps remaining "
          f"({steps} steps/epoch, checkpoint every "
          f"{config.NN1_TEXT_SAVE_EVERY_STEPS} steps)")
    run_loss = run_all_cos = run_content_cos = 0.0
    window_steps = 0
    pbar = tqdm(total=target_steps, initial=completed_steps,
                desc="text NN1", mininterval=2.0)
    try:
        for step_idx in range(completed_steps, target_steps):
            if stop_requested():
                raise KeyboardInterrupt
            nn1.train()
            inp, nxt = get_batch(train_stream, B, train_batch_rng)
            loss, all_cos, content_cos = step(inp, nxt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            completed_steps = step_idx + 1
            run_loss += loss.item()
            run_all_cos += all_cos
            run_content_cos += content_cos
            window_steps += 1
            pbar.update(1)
            if completed_steps % 50 == 0:
                pbar.set_postfix(
                    loss=f"{run_loss/window_steps:.4f}",
                    all_cos=f"{run_all_cos/window_steps:.4f}",
                    content_cos=f"{run_content_cos/window_steps:.4f}")
                run_loss = run_all_cos = run_content_cos = 0.0
                window_steps = 0
            if completed_steps % config.NN1_TEXT_SAVE_EVERY_STEPS == 0:
                save_checkpoint(False)
                pbar.write(
                    f"autosaved step {completed_steps} -> "
                    f"{config.NN1_TEXT_CKPT.name}")
            if completed_steps % steps == 0:
                nn1.train(False)
                with torch.no_grad():
                    inp_e, nxt_e = get_batch(eval_stream, B, eval_batch_rng)
                    ev_loss, ev_all_cos, ev_content_cos = step(inp_e, nxt_e)
                epoch = completed_steps // steps
                pbar.write(
                    f"epoch {epoch}  eval cos-dist {ev_loss:.4f}  "
                    f"all_cos {ev_all_cos:.4f}  "
                    f"content_cos {ev_content_cos:.4f}")
                record_metrics(config.RESULTS / "text_nn1.jsonl", {
                    "epoch": epoch,
                    "eval_all_cos": ev_all_cos,
                    "eval_content_cos": ev_content_cos,
                    "eval_cosine_distance": ev_loss.item(),
                    "run_id": config.RUN_ID,
                    "step": completed_steps,
                })
                save_checkpoint(completed_steps >= target_steps)
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

    save_checkpoint(True)
    print(f"completed {completed_steps} steps -> {config.NN1_TEXT_CKPT}")


if __name__ == "__main__":
    main()
