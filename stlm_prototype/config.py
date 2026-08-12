"""
Config for the STLM (Substrate-Translated Language Model) prototype.

    text -> substrate predictor (SP) -> substrate state or trace
         -> token readout (TR) -> token -> append, repeat

The original pilot uses a single-state token readout (TR-S). The completed
derived experiment trains a fresh causal token readout (TR-C), an ordinary
causal Transformer whose context consists only of the frozen SP's emitted
state sequence. The older residual history readout (TR-H) remains available as
a superseded diagnostic. Legacy NN1/NN2 constant and artifact names remain
below so the published checkpoints and their hashes stay compatible.
"""
from pathlib import Path
import os
import re

import torch


def env_int(name, default):
    """Read an integer override for unattended runs."""
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_float(name, default):
    """Read a floating-point override for unattended runs."""
    value = os.environ.get(name)
    return default if value is None else float(value)


def env_int_alias(primary, legacy, default):
    """Read a renamed integer option without silently accepting a conflict."""
    primary_value = os.environ.get(primary)
    legacy_value = os.environ.get(legacy)
    if primary_value is not None and legacy_value is not None:
        if int(primary_value) != int(legacy_value):
            raise ValueError(
                f"{primary} and legacy {legacy} disagree; set only {primary}")
    value = primary_value if primary_value is not None else legacy_value
    return default if value is None else int(value)


def env_run_id(name, default):
    """Read a filesystem-safe experiment identifier."""
    value = os.environ.get(name, default)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            f"{name} must contain only letters, numbers, '.', '_' or '-'")
    return value


def env_choice(name, default, choices):
    value = os.environ.get(name, default)
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}")
    return value

# --- paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

RUN_ID      = env_run_id("STLM_RUN_ID", "replication_seed42_e1_n2-100k")
EXPERIMENT_TYPE = env_choice(
    "STLM_EXPERIMENT_TYPE",
    "substrate",
    {"substrate", "direct_token", "trace_readout", "contextual_readout"},
)
SEED        = env_int("STLM_SEED", 42)
TRACE_ADAPTER_INIT_SEED = SEED
TRACE_CONTEXT_SAMPLING_SEED = SEED + 1_000
TRACE_DIAGNOSTIC_SHUFFLE_SEED = SEED + 3_000
RUN_DIR     = ROOT / "runs" / RUN_ID
CHECKPOINTS = RUN_DIR / "checkpoints"
RESULTS     = RUN_DIR / "results"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# Derived readout experiments consume causal substrate sequences emitted by
# frozen substrate predictors (SPs) from a published source run. Source
# checkpoints stay in their original run and are never copied into or modified
# by the derived run. TRACE_* settings below belong only to the superseded TR-H
# diagnostic; CONTEXTUAL_* settings define the corrected TR-C experiment.
SOURCE_RUN_ID      = env_run_id("STLM_SOURCE_RUN_ID", "data5x_seed42")
SOURCE_RUN_DIR     = ROOT / "runs" / SOURCE_RUN_ID
SOURCE_CHECKPOINTS = SOURCE_RUN_DIR / "checkpoints"
DIRECT_BASELINE_RUN_ID = env_run_id(
    "STLM_DIRECT_BASELINE_RUN_ID", "baseline5x_seed42")
DIRECT_BASELINE_CKPT = (
    ROOT / "runs" / DIRECT_BASELINE_RUN_ID
    / "checkpoints" / "baseline_direct_token.pt")
TRACE_LENGTH       = env_int("STLM_TRACE_LENGTH", 128)
if TRACE_LENGTH < 2:
    raise ValueError(
        "STLM_TRACE_LENGTH must be at least 2 "
        "(one history state plus the current state)")
TRACE_PIXEL_BATCH = env_int("STLM_TRACE_PIXEL_BATCH", 8)
TRACE_DESCRIPTION_BATCH = env_int("STLM_TRACE_DESCRIPTION_BATCH", 8)
TRACE_LR = env_float("STLM_TRACE_LR", 1e-3)
TRACE_SAVE_EVERY_STEPS = env_int("STLM_TRACE_SAVE_EVERY_STEPS", 250)
TRACE_STATE_DIM = env_int("STLM_TRACE_STATE_DIM", 128)
TRACE_TEMPORAL_FF_DIM = env_int("STLM_TRACE_TEMPORAL_FF_DIM", 256)
TRACE_TEMPORAL_HEADS = env_int("STLM_TRACE_TEMPORAL_HEADS", 4)
TRACE_PIXEL_GRID = env_int("STLM_TRACE_PIXEL_GRID", 8)
TRACE_DESCRIPTION_BINS = env_int("STLM_TRACE_DESCRIPTION_BINS", 8)
for name, value in {
    "STLM_TRACE_PIXEL_BATCH": TRACE_PIXEL_BATCH,
    "STLM_TRACE_DESCRIPTION_BATCH": TRACE_DESCRIPTION_BATCH,
    "STLM_TRACE_SAVE_EVERY_STEPS": TRACE_SAVE_EVERY_STEPS,
    "STLM_TRACE_STATE_DIM": TRACE_STATE_DIM,
    "STLM_TRACE_TEMPORAL_FF_DIM": TRACE_TEMPORAL_FF_DIM,
    "STLM_TRACE_TEMPORAL_HEADS": TRACE_TEMPORAL_HEADS,
    "STLM_TRACE_PIXEL_GRID": TRACE_PIXEL_GRID,
    "STLM_TRACE_DESCRIPTION_BINS": TRACE_DESCRIPTION_BINS,
}.items():
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
if TRACE_LR <= 0:
    raise ValueError("STLM_TRACE_LR must be positive")
if TRACE_STATE_DIM % TRACE_TEMPORAL_HEADS:
    raise ValueError(
        "STLM_TRACE_STATE_DIM must be divisible by "
        "STLM_TRACE_TEMPORAL_HEADS")

VOCAB_FILE      = DATA / "vocab.json"
DESCRIPTIONS    = DATA / "word_symbolic_descriptions.json"  # per-word LLM-authored prompts

# Phase-0 output layout (see phase0_canonicals.py for details):
#   data/<NATIVE>/<word>.png                 native-resolution PNG
#   data/<NATIVE>/downsampling/<word>.png    64x64 downsample (if NATIVE > 64)
#   data/canonicals.npz                      packed [VOCAB_SIZE, 3, 64, 64] uint8
CANONICALS_FILE = DATA / "canonicals.npz"       # phase 0 packed-tensor output
NN1_CKPT        = CHECKPOINTS / "nn1_pixel_predictor.pt"
NN2_CKPT        = CHECKPOINTS / "nn2_pixel_translator.pt"
SOURCE_PIXEL_SP_CKPT = SOURCE_CHECKPOINTS / "nn1_pixel_predictor.pt"
SOURCE_PIXEL_TR_S_CKPT = (
    SOURCE_CHECKPOINTS / "nn2_pixel_translator.pt")
TRACE_PIXEL_CKPT = CHECKPOINTS / "tr_h_pixel.pt"
CONTEXTUAL_PIXEL_CKPT = CHECKPOINTS / "tr_c_pixel.pt"

# --- device ------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- substrate (the per-position pixel image) --------------------------
IMG_SIZE   = 64                    # 64x64 RGB image per position
IMG_CH     = 3
VOCAB_SIZE = 1024

# --- Pixel SP (legacy NN1 names retained for checkpoint compatibility) -
NN1_TXT_LAYERS  = 6
NN1_TXT_D_MODEL = 384
NN1_TXT_HEADS   = 6
NN1_CONTEXT     = 128
if TRACE_LENGTH > NN1_CONTEXT:
    raise ValueError(
        f"STLM_TRACE_LENGTH cannot exceed the {NN1_CONTEXT}-token "
        "source context")

NN1_EPOCHS = env_int("STLM_PIXEL_NN1_EPOCHS", 1)  # total target, not additional
NN1_BATCH  = 16             # smaller batch -- per-position image is 12,288 floats
NN1_LR     = 3e-4
NN1_SAVE_EVERY_STEPS = env_int("STLM_PIXEL_NN1_SAVE_EVERY_STEPS", 250)

# --- Pixel TR-S (legacy NN2 names retained for compatibility) ----------
NN2_CHANNELS  = (32, 64, 128, 256)   # four conv blocks: 64 -> 32 -> 16 -> 8 -> 4
NN2_FC_HIDDEN = 256
NN2_BATCH     = 8
NN2_LR        = 1e-3
NN2_SAVE_EVERY_STEPS = env_int("STLM_PIXEL_NN2_SAVE_EVERY_STEPS", 250)

# --- text-embedding substrate -----------------------------------------
# Per-position per-token BERT-encoding substrate. Each word's canonical is
# the [TEXT_DESC_LEN, TEXT_DIM] sequence of per-subword BERT hidden states
# (NOT mean-pooled into one vector -- pooling would destroy word order and
# argument binding). The SP emits a unit-norm vector at each position; cosine
# loss is measured against the canonicals. TR-S reads the predicted sequence
# and emits a vocab token. This preserves spatial truth (order + binding)
# without paying the 4096-way categorical cross-entropy cost of explicit IDs.
TEXT_ENCODER_NAME    = "sentence-transformers/all-MiniLM-L6-v2"   # 22 MB, 384-dim hidden
TEXT_ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TEXT_DESC_LEN        = 56                                          # max BERT subword tokens per description
TEXT_DIM             = 384                                         # MiniLM hidden size
TEXT_CANONICALS_FILE = DATA / "text_token_canonicals.npz"
NN1_TEXT_CKPT        = CHECKPOINTS / "nn1_text_token_predictor.pt"
NN2_TEXT_CKPT        = CHECKPOINTS / "nn2_text_transformer_translator.pt"
SOURCE_DESCRIPTION_SP_CKPT = (
    SOURCE_CHECKPOINTS / "nn1_text_token_predictor.pt")
SOURCE_DESCRIPTION_TR_S_CKPT = (
    SOURCE_CHECKPOINTS / "nn2_text_transformer_translator.pt")
TRACE_DESCRIPTION_CKPT = CHECKPOINTS / "tr_h_description.pt"
CONTEXTUAL_DESCRIPTION_CKPT = CHECKPOINTS / "tr_c_description.pt"
NN1_TEXT_EPOCHS      = env_int("STLM_TEXT_NN1_EPOCHS", 1)          # total target
NN1_TEXT_BATCH       = 4                                           # smaller per-position output now -- can fit 2x
NN2_TEXT_BATCH       = 8
NN2_TEXT_D_MODEL     = 384
NN2_TEXT_HEADS       = 6
NN2_TEXT_LAYERS      = 1
NN2_TEXT_FF_DIM      = 1536
NN1_TEXT_SAVE_EVERY_STEPS = env_int("STLM_TEXT_NN1_SAVE_EVERY_STEPS", 1000)
NN2_TEXT_SAVE_EVERY_STEPS = env_int("STLM_TEXT_NN2_SAVE_EVERY_STEPS", 500)

# --- TR-C: ordinary causal token model over the full SP state sequence -----
# TR-C uses the existing 100,000 sampled-window budget and supervises all 128
# aligned positions, for 12.8M token targets. Record both quantities.
CONTEXTUAL_READOUT_INIT_SEED = SEED
CONTEXTUAL_CONTEXT_SAMPLING_SEED = SEED + 1_000
CONTEXTUAL_TRAIN_WINDOWS = env_int(
    "STLM_CONTEXTUAL_TRAIN_WINDOWS", 100_000)
CONTEXTUAL_PIXEL_BATCH = env_int("STLM_CONTEXTUAL_PIXEL_BATCH", 8)
CONTEXTUAL_DESCRIPTION_BATCH = env_int(
    "STLM_CONTEXTUAL_DESCRIPTION_BATCH", 8)
CONTEXTUAL_PIXEL_MICROBATCH = env_int(
    "STLM_CONTEXTUAL_PIXEL_MICROBATCH", 8)
CONTEXTUAL_DESCRIPTION_MICROBATCH = env_int(
    "STLM_CONTEXTUAL_DESCRIPTION_MICROBATCH", 8)
CONTEXTUAL_LR = env_float("STLM_CONTEXTUAL_LR", 3e-4)
CONTEXTUAL_SAVE_EVERY_STEPS = env_int(
    "STLM_CONTEXTUAL_SAVE_EVERY_STEPS", 250)
CONTEXTUAL_D_MODEL = env_int("STLM_CONTEXTUAL_D_MODEL", 256)
CONTEXTUAL_HEADS = env_int("STLM_CONTEXTUAL_HEADS", 8)
CONTEXTUAL_LAYERS = env_int("STLM_CONTEXTUAL_LAYERS", 2)
CONTEXTUAL_FF_DIM = env_int("STLM_CONTEXTUAL_FF_DIM", 1024)
CONTEXTUAL_PIXEL_POOL_GRID = env_int(
    "STLM_CONTEXTUAL_PIXEL_POOL_GRID", 8)
CONTEXTUAL_DESCRIPTION_POOL_BINS = env_int(
    "STLM_CONTEXTUAL_DESCRIPTION_POOL_BINS", 8)
for name, value in {
    "STLM_CONTEXTUAL_TRAIN_WINDOWS": CONTEXTUAL_TRAIN_WINDOWS,
    "STLM_CONTEXTUAL_PIXEL_BATCH": CONTEXTUAL_PIXEL_BATCH,
    "STLM_CONTEXTUAL_DESCRIPTION_BATCH": CONTEXTUAL_DESCRIPTION_BATCH,
    "STLM_CONTEXTUAL_PIXEL_MICROBATCH": CONTEXTUAL_PIXEL_MICROBATCH,
    "STLM_CONTEXTUAL_DESCRIPTION_MICROBATCH":
        CONTEXTUAL_DESCRIPTION_MICROBATCH,
    "STLM_CONTEXTUAL_SAVE_EVERY_STEPS": CONTEXTUAL_SAVE_EVERY_STEPS,
    "STLM_CONTEXTUAL_D_MODEL": CONTEXTUAL_D_MODEL,
    "STLM_CONTEXTUAL_HEADS": CONTEXTUAL_HEADS,
    "STLM_CONTEXTUAL_LAYERS": CONTEXTUAL_LAYERS,
    "STLM_CONTEXTUAL_FF_DIM": CONTEXTUAL_FF_DIM,
    "STLM_CONTEXTUAL_PIXEL_POOL_GRID": CONTEXTUAL_PIXEL_POOL_GRID,
    "STLM_CONTEXTUAL_DESCRIPTION_POOL_BINS":
        CONTEXTUAL_DESCRIPTION_POOL_BINS,
}.items():
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
if CONTEXTUAL_LR <= 0:
    raise ValueError("STLM_CONTEXTUAL_LR must be positive")
if CONTEXTUAL_PIXEL_MICROBATCH > CONTEXTUAL_PIXEL_BATCH:
    raise ValueError(
        "STLM_CONTEXTUAL_PIXEL_MICROBATCH cannot exceed its logical batch")
if CONTEXTUAL_DESCRIPTION_MICROBATCH > CONTEXTUAL_DESCRIPTION_BATCH:
    raise ValueError(
        "STLM_CONTEXTUAL_DESCRIPTION_MICROBATCH cannot exceed its logical "
        "batch")
if CONTEXTUAL_D_MODEL % CONTEXTUAL_HEADS:
    raise ValueError(
        "STLM_CONTEXTUAL_D_MODEL must be divisible by "
        "STLM_CONTEXTUAL_HEADS")

# --- corpus (phases 1 and 2) ------------------------------------------
CORPUS_NAME   = "roneneldan/TinyStories"
CORPUS_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
CORPUS_DATASET_FINGERPRINT = "050e3813ad82e8d9"
MAX_SENTENCES = env_int("STLM_MAX_SENTENCES", 50_000)
EVAL_CONTEXTS = 64
READOUT_TRAIN_TARGETS = env_int_alias(
    "STLM_READOUT_TRAIN_TARGETS", "STLM_NN2_TRAIN_TARGETS", 100_000)
READOUT_EVAL_TARGETS = env_int_alias(
    "STLM_READOUT_EVAL_TARGETS", "STLM_NN2_EVAL_TARGETS", 1_024)
# Legacy Python names remain the shared interface for the published scripts.
NN2_TRAIN_TARGETS = READOUT_TRAIN_TARGETS
NN2_EVAL_TARGETS = READOUT_EVAL_TARGETS

PUBLISHED_RUN_IDS = frozenset({
    "paper_seed42_e1_n2-100k",
    "data5x_seed42",
    "baseline5x_seed42",
    "tr_c_data5x_seed42_t128",
})


def require_experiment_type(expected):
    if EXPERIMENT_TYPE != expected:
        raise RuntimeError(
            f"this stage requires STLM_EXPERIMENT_TYPE={expected}, got "
            f"{EXPERIMENT_TYPE}")


def require_writable_run():
    if RUN_ID in PUBLISHED_RUN_IDS:
        raise RuntimeError(
            f"{RUN_ID} is a published run and is read-only; use a new "
            "STLM_RUN_ID for training or replication")

# Exact input artifacts used by the reported pilot. The runner verifies these
# before recording a new manifest; changing an input requires a new hash and
# run ID rather than silently reusing published provenance.
VOCAB_SHA256 = "fc33302323234254df4644f8db17a62a0b4f06719d4f0f8dc5ecb4e12a41995d"
DESCRIPTIONS_SHA256 = "0410efbb1e0719a08bb0e0a2df1df90a5e343ceebc9afdfad174ab021c17fb0b"
PIXEL_CANONICALS_SHA256 = "07eaf9a5908855bb2d100a9eb1f006df9deb35d1a2045800563db85fb936809e"
TEXT_CANONICALS_SHA256 = "0e95393075c51fb0dccee0034d00cf2c765ce65b41ec5558b63c5b0565d0c36b"
SANA_MODEL_NAME = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
SANA_MODEL_REVISION = "b3c9ce6f29ad4161a00fa58a62e476b9c75ca934"

# Retrospectively recovered and independently hashed from the immutable
# TinyStories revision above. These bind the two reported legacy runs to the
# exact token streams they consumed even though their original manifests
# predated corpus-identity fields.
LEGACY_RUN_CORPUS_IDENTITIES = {
    "paper_seed42_e1_n2-100k": {
        "max_sentences": 50_000,
        "token_count": 8_826_412,
        "token_stream_sha256_int64_le":
            "4c14f625dc8389df368c4982743baa9276cbec5202cdac30eee5de8f6eac9d17",
    },
    "data5x_seed42": {
        "max_sentences": 250_000,
        "token_count": 44_812_714,
        "token_stream_sha256_int64_le":
            "4ef516a821988476725ce7b3de3c906dfa66b783ba2481908e4943ec3406874f",
    },
}
