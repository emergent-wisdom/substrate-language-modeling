"""Read-only re-evaluation of the published pilot checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path


PUBLISHED_PATH = Path(__file__).parent / "published" / "pilot_runs.json"
PUBLISHED = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_id",
        choices=tuple(sorted(PUBLISHED["runs"])),
    )
    parser.add_argument(
        "--medium",
        choices=("pixel", "text", "both"),
        default="both",
    )
    parser.add_argument(
        "--loss-tolerance",
        type=float,
        default=1e-6,
        help="absolute cross-device tolerance for evaluation loss",
    )
    return parser.parse_args()


ARGS = parse_args()
RUN_RECORD = PUBLISHED["runs"][ARGS.run_id]

# config reads these at import time.
os.environ["STLM_EXPERIMENT_TYPE"] = RUN_RECORD.get(
    "experiment_type", "substrate")
os.environ["STLM_RUN_ID"] = ARGS.run_id
os.environ["STLM_MAX_SENTENCES"] = str(RUN_RECORD["max_sentences"])

import torch
import torch.nn.functional as F

import config
from baseline_direct_token import TokenPredictor
from corpus import load_corpus
from models import (DescriptionEmbeddingClassifier,
                    DescriptionEmbeddingPredictor, ImageClassifier,
                    ImagePredictor)
from training_utils import (last_token_eval_batches,
                            last_token_eval_targets, seed_everything,
                            verify_file_sha256)
from vocab import Tokenizer


DIRECT_CKPT = config.CHECKPOINTS / "baseline_direct_token.pt"


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=config.DEVICE)
    if checkpoint.get("run_id") != ARGS.run_id:
        raise RuntimeError(f"{path.name} belongs to another run")
    if not checkpoint.get("run_complete", False):
        raise RuntimeError(f"{path.name} is not marked complete")
    return checkpoint


def evaluate_pair(nn1, nn2, eval_stream):
    totals = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "n": 0}
    nn1.train(False)
    nn2.train(False)
    with torch.no_grad():
        batches = last_token_eval_batches(
            eval_stream,
            config.NN1_CONTEXT,
            config.NN2_BATCH,
            config.NN2_EVAL_TARGETS,
            config.DEVICE,
        )
        for inp, nxt in batches:
            substrate = nn1.forward_last(inp)
            logits = nn2(substrate)
            batch_size = nxt.shape[0]
            totals["loss"] += F.cross_entropy(
                logits, nxt, reduction="sum").item()
            pred = logits.topk(min(10, logits.shape[-1]), dim=-1).indices
            correct = pred.eq(nxt[:, None])
            totals["top1"] += correct[:, :1].any(dim=1).sum().item()
            totals["top5"] += correct[:, :5].any(dim=1).sum().item()
            totals["top10"] += correct.any(dim=1).sum().item()
            totals["n"] += batch_size
    n = totals["n"]
    return {
        "eval_loss": totals["loss"] / n,
        "eval_top1": totals["top1"] / n,
        "eval_top5": totals["top5"] / n,
        "eval_top10": totals["top10"] / n,
        "evaluation_examples": n,
    }


def evaluate_direct(model, eval_stream):
    totals = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "n": 0}
    model.train(False)
    with torch.no_grad():
        batches = last_token_eval_batches(
            eval_stream,
            config.NN1_CONTEXT,
            config.NN2_BATCH,
            config.NN2_EVAL_TARGETS,
            config.DEVICE,
        )
        for inp, nxt in batches:
            logits = model.forward_last(inp)
            batch_size = nxt.shape[0]
            totals["loss"] += F.cross_entropy(
                logits, nxt, reduction="sum").item()
            pred = logits.topk(min(10, logits.shape[-1]), dim=-1).indices
            correct = pred.eq(nxt[:, None])
            totals["top1"] += correct[:, :1].any(dim=1).sum().item()
            totals["top5"] += correct[:, :5].any(dim=1).sum().item()
            totals["top10"] += correct.any(dim=1).sum().item()
            totals["n"] += batch_size
    n = totals["n"]
    loss = totals["loss"] / n
    return {
        "eval_loss": loss,
        "eval_perplexity": math.exp(loss),
        "eval_top1": totals["top1"] / n,
        "eval_top5": totals["top5"] / n,
        "eval_top10": totals["top10"] / n,
        "evaluation_examples": n,
    }


def frequency_metrics(eval_stream, tok):
    targets = last_token_eval_targets(
        eval_stream, config.NN1_CONTEXT, config.NN2_EVAL_TARGETS)
    counts = torch.bincount(targets, minlength=len(tok))
    values = counts.topk(10).values
    total = targets.numel()
    return {
        "frequency_top1": values[:1].sum().item() / total,
        "frequency_top5": values[:5].sum().item() / total,
        "frequency_top10": values.sum().item() / total,
    }


def pixel_models():
    nn1_ck = load_checkpoint(config.NN1_CKPT)
    nn2_ck = load_checkpoint(config.NN2_CKPT)
    nn1 = ImagePredictor(
        n_tokens=nn1_ck["n_tokens"],
        img_size=nn1_ck["img_size"],
        img_ch=nn1_ck["img_ch"],
        context_len=nn1_ck["context_len"],
        txt_layers=nn1_ck["txt_layers"],
        txt_d_model=nn1_ck["txt_d_model"],
        txt_heads=nn1_ck["txt_heads"],
    )
    nn2 = ImageClassifier(
        nn2_ck["img_size"],
        nn2_ck["n_tokens"],
        channels=tuple(nn2_ck["channels"]),
        fc_hidden=nn2_ck["fc_hidden"],
    )
    nn1.load_state_dict(nn1_ck["state_dict"])
    nn2.load_state_dict(nn2_ck["state_dict"])
    return nn1.to(config.DEVICE), nn2.to(config.DEVICE)


def text_models():
    nn1_ck = load_checkpoint(config.NN1_TEXT_CKPT)
    nn2_ck = load_checkpoint(config.NN2_TEXT_CKPT)
    nn1 = DescriptionEmbeddingPredictor(
        n_tokens=nn1_ck["n_tokens"],
        desc_len=nn1_ck["desc_len"],
        embed_dim=nn1_ck["embed_dim"],
        context_len=nn1_ck["context_len"],
        txt_layers=nn1_ck["txt_layers"],
        txt_d_model=nn1_ck["txt_d_model"],
        txt_heads=nn1_ck["txt_heads"],
    )
    nn2 = DescriptionEmbeddingClassifier(
        embed_dim=nn2_ck["embed_dim"],
        n_tokens=nn2_ck["n_tokens"],
        desc_len=nn2_ck["desc_len"],
        d_model=nn2_ck["d_model"],
        n_heads=nn2_ck["n_heads"],
        n_layers=nn2_ck["n_layers"],
        ff_dim=nn2_ck["ff_dim"],
    )
    nn1.load_state_dict(nn1_ck["state_dict"])
    nn2.load_state_dict(nn2_ck["state_dict"])
    return nn1.to(config.DEVICE), nn2.to(config.DEVICE)


def direct_model():
    ck = load_checkpoint(DIRECT_CKPT)
    model = TokenPredictor(
        n_tokens=ck["n_tokens"],
        context_len=ck["context_len"],
        txt_layers=ck["txt_layers"],
        txt_d_model=ck["txt_d_model"],
        txt_heads=ck["txt_heads"],
    )
    model.load_state_dict(ck["state_dict"])
    return model.to(config.DEVICE), ck


def verify_metrics(record_key, label, actual):
    expected = RUN_RECORD["metrics"][record_key]
    mismatches = []
    for key, value in actual.items():
        if key not in expected:
            continue
        if key == "eval_loss":
            tolerance = ARGS.loss_tolerance
        elif key == "eval_perplexity":
            tolerance = ARGS.loss_tolerance * expected[key]
        else:
            tolerance = 0.0
        if abs(value - expected[key]) > tolerance:
            mismatches.append(
                f"{key}: expected {expected[key]!r}, got {value!r}")
    if mismatches:
        raise RuntimeError(
            f"{label} checkpoint verification failed: "
            + "; ".join(mismatches))
    print(f"{label}: PASS {json.dumps(actual, sort_keys=True)}")


def verify_checkpoint_hashes(medium):
    names = (
        ("nn1_pixel_predictor.pt", config.NN1_CKPT),
        ("nn2_pixel_translator.pt", config.NN2_CKPT),
    ) if medium == "pixel" else (
        ("nn1_text_token_predictor.pt", config.NN1_TEXT_CKPT),
        ("nn2_text_transformer_translator.pt", config.NN2_TEXT_CKPT),
    )
    for name, path in names:
        verify_file_sha256(
            path,
            RUN_RECORD["checkpoint_sha256"][name],
            f"{ARGS.run_id}/{name}",
        )


def verify_direct_checkpoint_hash():
    verify_file_sha256(
        DIRECT_CKPT,
        RUN_RECORD["checkpoint_sha256"]["baseline_direct_token.pt"],
        f"{ARGS.run_id}/baseline_direct_token.pt",
    )


def verify_small_artifact_hashes():
    for relative_path, expected_hash in RUN_RECORD.get(
            "artifact_sha256", {}).items():
        verify_file_sha256(
            config.RUN_DIR / relative_path,
            expected_hash,
            f"{ARGS.run_id}/{relative_path}",
        )


def main():
    seed_everything(config.SEED)
    tok = Tokenizer()
    stream = load_corpus(tok)
    n_eval = config.NN1_CONTEXT * config.EVAL_CONTEXTS
    eval_stream = stream[-n_eval:]
    frequencies = frequency_metrics(eval_stream, tok)

    if config.EXPERIMENT_TYPE == "direct_token":
        if ARGS.medium != "both":
            raise RuntimeError("--medium applies only to substrate runs")
        verify_small_artifact_hashes()
        verify_direct_checkpoint_hash()
        model, checkpoint = direct_model()
        metrics = evaluate_direct(model, eval_stream)
        metrics.update({
            "params": model.n_params(),
            "head_params": sum(
                parameter.numel() for parameter in model.head.parameters()),
            "step": checkpoint["steps_trained"],
            **frequencies,
        })
        verify_metrics("direct_token", "direct-token", metrics)
        return

    media = ("pixel", "text") if ARGS.medium == "both" else (ARGS.medium,)
    for medium in media:
        verify_checkpoint_hashes(medium)
        nn1, nn2 = pixel_models() if medium == "pixel" else text_models()
        metrics = evaluate_pair(nn1, nn2, eval_stream)
        metrics.update(frequencies)
        verify_metrics(f"{medium}_nn2", medium, metrics)
        del nn1, nn2
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
