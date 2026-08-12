"""
phase1_nn1.py -- train NN1, the text-to-pixel-image regressor.

For each text position, the target is the canonical 3x64x64 pixel image
of the next token (from canonicals.npz). Loss is per-pixel MSE. NN2 is
not involved.

    python phase1_nn1.py     # needs data/canonicals.npz
"""
import numpy as np
import torch
import torch.nn.functional as F

import config
from corpus import load_corpus
from models import ImagePredictor
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
    if not config.CANONICALS_FILE.exists():
        raise FileNotFoundError("run phase0_canonicals.py first")
    verify_file_sha256(
        config.CANONICALS_FILE,
        config.PIXEL_CANONICALS_SHA256,
        "pixel canonicals",
    )

    canonicals_np = np.load(config.CANONICALS_FILE)["canonicals"]
    canonicals = (torch.from_numpy(canonicals_np).float()
                  .to(config.DEVICE) / 255.0)
    # [VOCAB_SIZE, 3, IMG_SIZE, IMG_SIZE] in [0, 1]

    nn1 = ImagePredictor(
        n_tokens=len(tok),
        img_size=config.IMG_SIZE, img_ch=config.IMG_CH,
        context_len=config.NN1_CONTEXT,
        txt_layers=config.NN1_TXT_LAYERS, txt_d_model=config.NN1_TXT_D_MODEL,
        txt_heads=config.NN1_TXT_HEADS).to(config.DEVICE)
    print(f"NN1 {nn1.n_params() / 1e6:.1f}M params on {config.DEVICE}")
    opt = torch.optim.AdamW(nn1.parameters(), lr=config.NN1_LR)

    stream = load_corpus(tok)
    T, B = config.NN1_CONTEXT, config.NN1_BATCH
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

    def step_loss(inp, nxt):
        pred = nn1(inp)                                # [B, T, 3, H, W]
        tgt = canonicals[nxt]                          # [B, T, 3, H, W]
        return F.mse_loss(pred, tgt)

    steps = max(1, len(train_stream) // (B * T))
    completed_steps = 0
    target_steps = config.NN1_EPOCHS * steps
    if config.NN1_CKPT.exists():
        prev = torch.load(config.NN1_CKPT, map_location=config.DEVICE)
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
        print(f"resumed from {config.NN1_CKPT.name}: "
              f"{completed_steps}/{target_steps} steps trained")
    else:
        print("no existing pixel NN1 checkpoint -- training from scratch")

    def save_checkpoint(done):
        torch.save({"state_dict": nn1.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
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
                    "img_size": config.IMG_SIZE, "img_ch": config.IMG_CH,
                    "context_len": config.NN1_CONTEXT,
                    "txt_layers": config.NN1_TXT_LAYERS,
                    "txt_d_model": config.NN1_TXT_D_MODEL,
                    "txt_heads": config.NN1_TXT_HEADS}, config.NN1_CKPT)

    print(f"{target_steps - completed_steps} steps remaining "
          f"({steps} steps/epoch, checkpoint every "
          f"{config.NN1_SAVE_EVERY_STEPS} steps)")
    run_loss = 0.0
    window_steps = 0
    try:
        for step_idx in range(completed_steps, target_steps):
            if stop_requested():
                raise KeyboardInterrupt
            nn1.train()
            inp, nxt = get_batch(train_stream, B, train_batch_rng)
            loss = step_loss(inp, nxt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            completed_steps = step_idx + 1
            run_loss += loss.item()
            window_steps += 1

            if completed_steps % 50 == 0:
                print(f"step {completed_steps}/{target_steps}  "
                      f"train MSE {run_loss / window_steps:.5f}")
                run_loss = 0.0
                window_steps = 0
            if completed_steps % config.NN1_SAVE_EVERY_STEPS == 0:
                save_checkpoint(False)
                print(f"autosaved step {completed_steps} -> "
                      f"{config.NN1_CKPT.name}")
            if completed_steps % steps == 0:
                nn1.train(False)
                with torch.no_grad():
                    inp_e, nxt_e = get_batch(eval_stream, B, eval_batch_rng)
                    ev = step_loss(inp_e, nxt_e).item()
                print(f"epoch {completed_steps // steps}  eval MSE {ev:.5f}")
                record_metrics(config.RESULTS / "pixel_nn1.jsonl", {
                    "epoch": completed_steps // steps,
                    "eval_mse": ev,
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
        print(f"interrupted at step {completed_steps}; checkpoint saved")
        raise SystemExit(130)

    save_checkpoint(True)
    print(f"completed {completed_steps} steps -> {config.NN1_CKPT}")


if __name__ == "__main__":
    main()
