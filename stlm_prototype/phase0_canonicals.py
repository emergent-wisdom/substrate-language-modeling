"""
phase0_canonicals.py -- render one 64x64 RGB canonical per word via
Sana-Sprint, using LLM-authored symbolic descriptions.

Folder structure (per --native resolution N):
  data/N/<word>.png                  native-resolution Sana output
  data/N/downsampling/<word>.png     downsampled to IMG_SIZE (64x64) -- only
                                     when N > IMG_SIZE; the substrate the
                                     training pipeline reads from.

For each word in the vocabulary:
  - Special tokens (<bos>, <eos>, <unk>) get solid placeholder colors
    written at IMG_SIZE directly (no Sana call).
  - Content words get rendered from their description in
    `data/word_symbolic_descriptions.json` by Sana-Sprint 1.6B at
    N x N, then (if N > IMG_SIZE) downsampled to IMG_SIZE.

Resumable: skips any word whose native-resolution PNG already exists.

    python phase0_canonicals.py --native 512        # all 1024 words at 512
    python phase0_canonicals.py --n 10 --native 64  # first 10 at native 64
    python phase0_canonicals.py --words apple,dog,the,but --native 1024

Output:
  data/<N>/<word>.png                per-word native PNG
  data/<N>/downsampling/<word>.png   per-word 64x64 PNG (if N > 64)
  data/canonicals.npz                packed [VOCAB_SIZE, 3, 64, 64] uint8
                                     (written when the full set is complete)
"""
import argparse
import json
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import config
from vocab import Tokenizer

# --- Sana-Sprint settings ----------------------------------------------
SANA_MODEL = config.SANA_MODEL_NAME
SANA_MODEL_REVISION = config.SANA_MODEL_REVISION
SANA_STEPS = 2                # Sana-Sprint requires 2 in diffusers 0.38
SANA_GUIDANCE = 4.5
SANA_NATIVE = 512             # default Sana render resolution
# Special-token canonical colors (RGB grayscale value)
SPECIAL_COLORS = {"<bos>": 255, "<eos>": 0, "<unk>": 128}


def native_dir(native):
    return config.DATA / str(native)


def downsampling_dir(native):
    return native_dir(native) / "downsampling"


def load_sana():
    """Load Sana-Sprint pipeline on MPS in bfloat16."""
    print(f"loading {SANA_MODEL} ...")
    from diffusers import SanaSprintPipeline
    pipe = SanaSprintPipeline.from_pretrained(
        SANA_MODEL,
        revision=SANA_MODEL_REVISION,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(config.DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def render_one(pipe, prompt, native, steps):
    """Render `prompt` at native x native. Returns (native_arr, ds_arr).

    native_arr: [native, native, 3] uint8
    ds_arr:     [IMG_SIZE, IMG_SIZE, 3] uint8, downsampled via LANCZOS
                (equals native_arr when native == IMG_SIZE)
    """
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            height=native, width=native,
            num_inference_steps=steps,
            guidance_scale=SANA_GUIDANCE,
        )
    native_img = result.images[0].convert("RGB")
    native_arr = np.asarray(native_img, dtype=np.uint8)
    if native == config.IMG_SIZE:
        return native_arr, native_arr
    ds_img = native_img.resize(
        (config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)
    ds_arr = np.asarray(ds_img, dtype=np.uint8)
    return native_arr, ds_arr


def solid_canonical(brightness):
    """Solid color [IMG_SIZE, IMG_SIZE, 3] uint8 for special tokens."""
    return np.full((config.IMG_SIZE, config.IMG_SIZE, 3),
                   brightness, dtype=np.uint8)


def ds_path_for(native, word):
    """Where the 64x64 canonical for `word` lives (varies with native)."""
    if native == config.IMG_SIZE:
        return native_dir(native) / f"{word}.png"
    return downsampling_dir(native) / f"{word}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="generate only the first N words")
    parser.add_argument("--words", type=str, default=None,
                        help="comma-separated words to (re)generate")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing PNGs")
    parser.add_argument("--native", type=int, default=SANA_NATIVE,
                        help=f"Sana render resolution (default {SANA_NATIVE})")
    parser.add_argument("--steps", type=int, default=SANA_STEPS,
                        help=f"Sana inference steps (default {SANA_STEPS})")
    args = parser.parse_args()

    tok = Tokenizer()
    if not config.DESCRIPTIONS.exists():
        raise FileNotFoundError(
            f"{config.DESCRIPTIONS} missing -- description file required")
    descs = json.loads(config.DESCRIPTIONS.read_text())
    print(f"loaded {len(descs)} descriptions from {config.DESCRIPTIONS.name}")

    # Set up output directories for this native resolution
    nd = native_dir(args.native)
    nd.mkdir(parents=True, exist_ok=True)
    if args.native != config.IMG_SIZE:
        downsampling_dir(args.native).mkdir(parents=True, exist_ok=True)
    print(f"output -> {nd}"
          f"{' (+downsampling/)' if args.native != config.IMG_SIZE else ''}")

    # Determine word set to render this run
    if args.words:
        wanted = [w.strip() for w in args.words.split(",") if w.strip()]
        targets = [(tok.word2id[w], w) for w in wanted if w in tok.word2id]
    else:
        words = tok.words[:args.n] if args.n else tok.words
        targets = list(enumerate(words))

    # Build to-do list (resumable -- skip existing unless --force).
    # Native PNG is the source of truth; if it exists, we have everything.
    todo = []
    for tid, w in targets:
        native_path = nd / f"{w}.png"
        if native_path.exists() and not args.force:
            continue
        todo.append((tid, w, native_path))

    pipe = None
    if not todo:
        print("all targets already rendered (use --force to overwrite)")
    else:
        print(f"to render: {len(todo)} / {len(targets)} words")
        t0 = time.time()
        for tid, w, native_path in tqdm(todo, desc="rendering"):
            if w in SPECIAL_COLORS:
                ds_arr = solid_canonical(SPECIAL_COLORS[w])
                native_arr = ds_arr  # specials are saved at IMG_SIZE everywhere
            else:
                if pipe is None:
                    pipe = load_sana()
                desc = descs.get(w, w)   # fallback to the word itself
                native_arr, ds_arr = render_one(
                    pipe, desc, args.native, args.steps)
            Image.fromarray(native_arr).save(native_path)
            if args.native != config.IMG_SIZE:
                Image.fromarray(ds_arr).save(ds_path_for(args.native, w))
        dt = time.time() - t0
        per = dt / max(1, len(todo))
        print(f"rendered {len(todo)} in {dt:.1f}s  ({per:.1f}s/image)")

    # Pack the 64x64 npz when the full vocabulary is present at this native
    canonicals = np.zeros(
        (len(tok), config.IMG_CH, config.IMG_SIZE, config.IMG_SIZE),
        dtype=np.uint8)
    n_have = 0
    for i, w in enumerate(tok.words):
        p = ds_path_for(args.native, w)
        if p.exists():
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            canonicals[i] = arr.transpose(2, 0, 1)
            n_have += 1

    if n_have == len(tok):
        np.savez_compressed(config.CANONICALS_FILE, canonicals=canonicals)
        print(f"saved packed tensor -> {config.CANONICALS_FILE}")
    else:
        print(f"have {n_have}/{len(tok)} 64x64 PNGs at native={args.native}; "
              f"npz will be written when the full set is rendered")


if __name__ == "__main__":
    main()
