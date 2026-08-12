"""Pinned TinyStories loading and fail-closed token-stream verification."""
import hashlib
import json
import sys

import torch
from tqdm import tqdm

import config


def token_stream_sha256(stream):
    """Hash the contiguous little-endian int64 token stream without copying."""
    array = stream.contiguous().numpy()
    if sys.byteorder != "little":
        array = array.byteswap()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def load_corpus(tok, verify_manifest=True, return_identity=False):
    """Load, tokenize, fingerprint, and optionally verify this run's corpus."""
    from datasets import load_dataset

    print(f"loading {config.CORPUS_NAME}@{config.CORPUS_REVISION[:12]} ...")
    ds = load_dataset(
        config.CORPUS_NAME,
        revision=config.CORPUS_REVISION,
        split="train",
    )
    dataset_fingerprint = getattr(ds, "_fingerprint", None)
    if dataset_fingerprint != config.CORPUS_DATASET_FINGERPRINT:
        raise RuntimeError(
            "TinyStories fingerprint mismatch: expected "
            f"{config.CORPUS_DATASET_FINGERPRINT}, got "
            f"{dataset_fingerprint!r}. Refusing to train on unverified data.")

    n = min(len(ds), config.MAX_SENTENCES)
    stream_ids, unk = [], 0
    for i, ex in enumerate(tqdm(ds, total=n, desc="tokenizing")):
        if i >= n:
            break
        ids = tok.encode(ex["text"], add_bos=True, add_eos=True)
        unk += sum(token_id == tok.unk for token_id in ids)
        stream_ids.extend(ids)

    stream = torch.tensor(stream_ids, dtype=torch.long)
    identity = {
        "dataset_revision": config.CORPUS_REVISION,
        "dataset_fingerprint": dataset_fingerprint,
        "split": "train",
        "max_sentences": config.MAX_SENTENCES,
        "records_consumed": n,
        "token_count": stream.numel(),
        "token_stream_sha256_int64_le": token_stream_sha256(stream),
    }
    print(f"corpus: {stream.numel():,} tokens "
          f"({100 * unk / max(stream.numel(), 1):.1f}% <unk>)")
    print(f"token stream sha256: "
          f"{identity['token_stream_sha256_int64_le']}")

    legacy = config.LEGACY_RUN_CORPUS_IDENTITIES.get(config.RUN_ID)
    if legacy is not None:
        _require_matching_fields(identity, legacy, "legacy corpus identity")

    if verify_manifest:
        _verify_manifest(identity)
    if return_identity:
        return stream, identity
    return stream


def _require_matching_fields(actual, expected, label):
    mismatches = {
        key: (expected_value, actual.get(key))
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    if mismatches:
        details = "; ".join(
            f"{key}: expected {want!r}, got {got!r}"
            for key, (want, got) in sorted(mismatches.items()))
        raise RuntimeError(f"{label} mismatch ({details})")


def _verify_manifest(identity):
    manifest_path = config.RUN_DIR / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} is missing. Run write_run_manifest.py before "
            "training so the input identity is recorded.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("experiment", {}).get("corpus_identity")
    if recorded is not None:
        _require_matching_fields(identity, recorded, "manifest corpus identity")
        return
    if config.RUN_ID not in config.LEGACY_RUN_CORPUS_IDENTITIES:
        raise RuntimeError(
            "manifest predates corpus fingerprints and this is not a recorded "
            "legacy run; use a new STLM_RUN_ID and create a new manifest")
