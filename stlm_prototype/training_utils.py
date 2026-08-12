"""Small reproducibility and checkpoint helpers shared by training stages."""
import hashlib
import json
import random
import signal
from pathlib import Path

import numpy as np
import torch

_STOP_REQUESTED = False


def seed_everything(seed):
    # Fail instead of silently using a known nondeterministic PyTorch kernel.
    # This constrains a fixed environment; it does not promise identical
    # floating-point results across devices or PyTorch releases.
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def capture_torch_rng():
    state = {"cpu": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_torch_rng(state):
    if not state:
        return
    # torch.load(map_location="mps") moves these byte tensors with the model.
    # Generator state APIs require CPU byte tensors even for the MPS generator.
    torch.set_rng_state(state["cpu"].cpu())
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def install_interrupt_handler():
    """Defer Ctrl-C/SIGTERM until the current optimizer step is complete."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = False

    def request_stop(_signum, _frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def stop_requested():
    return _STOP_REQUESTED


def fixed_eval_starts(data, context_len, n_targets):
    """Choose deterministic, evenly spaced next-token contexts."""
    n_available = len(data) - context_len
    n = min(n_targets, n_available)
    if n == n_available:
        return list(range(n_available))
    return torch.linspace(0, n_available - 1, steps=n).long().tolist()


def last_token_eval_batches(data, context_len, batch_size, n_targets, device):
    starts = fixed_eval_starts(data, context_len, n_targets)
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset:offset + batch_size]
        inp = torch.stack(
            [data[i:i + context_len] for i in batch_starts]).to(device)
        nxt = torch.stack([data[i + context_len] for i in batch_starts]).to(device)
        yield inp, nxt


def last_token_eval_targets(data, context_len, n_targets):
    starts = fixed_eval_starts(data, context_len, n_targets)
    return torch.stack([data[i + context_len] for i in starts])


def validate_checkpoint(checkpoint, run_id, seed, target_steps):
    if checkpoint.get("run_id") != run_id:
        raise RuntimeError("checkpoint belongs to a different STLM run")
    if checkpoint.get("seed") != seed:
        raise RuntimeError("checkpoint seed does not match this STLM run")
    saved_target = checkpoint.get("target_steps")
    if saved_target != target_steps:
        raise RuntimeError(
            f"checkpoint target is {saved_target} steps, current target is "
            f"{target_steps}; use a new STLM_RUN_ID for a different budget")


def record_metrics(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path, expected, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{label} is missing at {path}; fetch the published artifact inputs before "
            "creating or resuming a run")
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, got {actual}")
    print(f"verified {label} sha256: {actual}")
    return actual
