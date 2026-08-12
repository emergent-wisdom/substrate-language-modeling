"""
phase0_text_canonicals.py -- build text-embedding substrate canonicals.

For each vocabulary word, encode the LLM-authored description through a
frozen BERT-family text encoder and KEEP THE PER-TOKEN HIDDEN STATES
(no pooling). The substrate is a [TEXT_DESC_LEN, TEXT_DIM] sequence per
word -- positional order and attention-mixed binding are preserved by
construction, without paying a categorical-cross-entropy cost.

Specials get encoded sentinel strings ("beginning of sequence", etc).

    python phase0_text_canonicals.py

Outputs:
  data/text_token_canonicals.npz
    canonicals: [VOCAB_SIZE, TEXT_DESC_LEN, TEXT_DIM] float32
                every per-position vector is L2-normalized. Valid positions
                carry the BERT subword's hidden state; padded positions all
                carry the same fixed PAD vector (BERT's [PAD] input
                embedding, L2-normalized) so NN1 has a supervised target
                at every slot during phase 1.
    mask: [VOCAB_SIZE, TEXT_DESC_LEN] int64
          1 where the position is a real BERT subword token, 0 for padding.
          Kept purely for diagnostics (loss no longer masks anything).
"""
import json
import time

import numpy as np
import torch

import config
from vocab import SPECIALS, Tokenizer

SPECIAL_SENTINEL_STRINGS = {
    "<bos>": "beginning of sequence",
    "<eos>": "end of sequence",
    "<unk>": "unknown token",
}


def main():
    tok = Tokenizer()
    if not config.DESCRIPTIONS.exists():
        raise FileNotFoundError(
            f"{config.DESCRIPTIONS} missing -- description file required")
    descs = json.loads(config.DESCRIPTIONS.read_text())
    print(f"loaded {len(descs)} descriptions from {config.DESCRIPTIONS.name}")

    # Build per-word text list in vocab order.
    texts = []
    for w in tok.words:
        if w in SPECIALS:
            texts.append(SPECIAL_SENTINEL_STRINGS[w])
        else:
            texts.append(descs.get(w, w))
    print(f"texts to encode: {len(texts)} "
          f"({len(SPECIAL_SENTINEL_STRINGS)} sentinels + "
          f"{len(texts) - len(SPECIAL_SENTINEL_STRINGS)} descriptions)")

    print(f"loading encoder {config.TEXT_ENCODER_NAME} ...")
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    enc_tok = AutoTokenizer.from_pretrained(
        config.TEXT_ENCODER_NAME,
        revision=config.TEXT_ENCODER_REVISION,
    )
    enc_model = AutoModel.from_pretrained(
        config.TEXT_ENCODER_NAME,
        revision=config.TEXT_ENCODER_REVISION,
    )
    enc_model = enc_model.to(config.DEVICE)
    enc_model.train(False)
    n_enc_params = sum(p.numel() for p in enc_model.parameters())
    out_dim = enc_model.config.hidden_size
    print(f"  loaded in {time.time()-t0:.1f}s "
          f"({n_enc_params/1e6:.1f}M params, hidden dim {out_dim})")
    if out_dim != config.TEXT_DIM:
        raise RuntimeError(
            f"encoder hidden ({out_dim}) does not match config.TEXT_DIM "
            f"({config.TEXT_DIM})")

    print(f"encoding {len(texts)} texts (per-token hidden states) ...")
    t0 = time.time()
    BATCH = 32
    hidden_chunks = []
    mask_chunks = []
    n_truncated = 0

    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        # Padding to TEXT_DESC_LEN; truncation if longer.
        enc = enc_tok(
            batch,
            padding="max_length",
            max_length=config.TEXT_DESC_LEN,
            truncation=True,
            return_tensors="pt")
        # Track truncation: re-tokenize without padding/truncation and see if
        # any natural length > TEXT_DESC_LEN.
        for txt in batch:
            full = enc_tok(txt, return_tensors="pt").input_ids
            if full.shape[1] > config.TEXT_DESC_LEN:
                n_truncated += 1
        enc = {k: v.to(config.DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = enc_model(**enc)
        hidden_chunks.append(out.last_hidden_state.cpu().numpy())  # [B, L, D]
        mask_chunks.append(enc["attention_mask"].cpu().numpy())     # [B, L]

    hidden = np.concatenate(hidden_chunks, axis=0).astype(np.float32)
    mask = np.concatenate(mask_chunks, axis=0).astype(np.int64)
    print(f"  encoded in {time.time()-t0:.1f}s, "
          f"hidden {hidden.shape}, mask {mask.shape}")

    # Compute the fixed PAD vector: BERT's [PAD] token's input embedding,
    # L2-normalized. This becomes the canonical target at every padded slot
    # for every word, so NN1 can be trained at all 56 positions consistently.
    pad_token_id = enc_tok.pad_token_id
    if pad_token_id is None:
        raise RuntimeError("encoder tokenizer has no pad_token_id")
    with torch.no_grad():
        pad_input_embed = enc_model.embeddings.word_embeddings(
            torch.tensor([pad_token_id], device=config.DEVICE)).cpu().numpy()[0]
    pad_vector = pad_input_embed / (np.linalg.norm(pad_input_embed) + 1e-12)
    pad_vector = pad_vector.astype(np.float32)
    print(f"pad vector: norm {np.linalg.norm(pad_vector):.6f} "
          f"(from BERT [PAD] token id {pad_token_id})")

    # L2-normalize valid positions; fill padded positions with the fixed
    # PAD vector. Every position now has a unit-norm canonical target.
    for i in range(hidden.shape[0]):
        valid = mask[i].astype(bool)
        if valid.any():
            v = hidden[i][valid]
            norms = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
            hidden[i][valid] = v / norms
        hidden[i][~valid] = pad_vector

    # Sanity check: every position is unit-norm.
    all_norms = np.linalg.norm(hidden, axis=-1)
    assert np.allclose(all_norms, 1.0, atol=1e-5), \
        f"positions not unit-norm: range [{all_norms.min()}, {all_norms.max()}]"

    print(f"canonicals shape: {hidden.shape}")
    print(f"truncated descriptions: {n_truncated}/{len(tok)}")
    print(f"average valid length: {mask.sum(axis=1).mean():.1f}/{config.TEXT_DESC_LEN}")
    print(f"padded slots: {(1 - mask).sum()} total positions filled with PAD vector")

    np.savez_compressed(
        config.TEXT_CANONICALS_FILE,
        canonicals=hidden,
        mask=mask,
    )
    print(f"saved -> {config.TEXT_CANONICALS_FILE}")


if __name__ == "__main__":
    main()
