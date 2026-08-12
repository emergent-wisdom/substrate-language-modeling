# STLM prototype

A laptop-scale implementation of the **Substrate-Translated Language Model**, the
architecture described in `../substratelm.tex`. A **substrate predictor (SP)**
maps text context to substrate states, and a **token readout (TR)** maps those
states back to vocabulary logits. The two tested substrates are literal
64x64 RGB pixel images and per-token description embeddings.

## Loop

      text  ->  SP (substrate predictor)
                    |
                    v
            substrate state or trace
                    |
                    v
              TR (token readout)  ->  token  ->  append, repeat

- **SP** -- a causal transformer that emits one substrate state per position.
- **TR-S** -- the published single-state token readout. It reads only the
  current SP state.
- **TR-C** -- the causal token readout. A fresh medium-specific state adapter,
  shared across positions within that medium, maps every frozen-SP state to
  256 dimensions. A two-layer, eight-head causal Transformer reads the complete
  128-state sequence including the current state and emits vocabulary logits at
  every position. It never receives source-token IDs or SP hidden states.
- **Pixel states retain an explicit spatial axis.** The dense SP objective does
  not force its computation to respect locality, so this pilot tests a
  target-shaped interface rather than the paper's stronger rendering claim.

Legacy script and checkpoint filenames retain `nn1` and `nn2` so the published
artifacts and their hashes do not change.

## Completed five-condition comparison

The reported comparison evaluates five configurations on the same fixed
1,024-context holdout:

| Configuration | NLL | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|
| Direct-token LM | 2.208 | 44.7% | 74.6% | 85.3% |
| Description SP + TR-C | 2.616 | 41.9% | 68.6% | 78.0% |
| Pixel SP + TR-C | 2.954 | 37.8% | 61.6% | 72.2% |
| Description SP + TR-S | 3.541 | 35.0% | 54.2% | 62.8% |
| Pixel SP + TR-S | 4.283 | 27.8% | 44.3% | 52.9% |

The completed run `tr_c_data5x_seed42_t128` reuses and freezes both SPs
from `data5x_seed42`, freshly initializes one complete TR-C per medium, and
re-evaluates the two existing TR-S conditions and `baseline5x_seed42`. Neither
TR-S is a component or initializer of TR-C.

Each TR-C saw 100,000 deterministically sampled length-128 windows shared
across media, with same-index cross-entropy at all positions: 12.8 million
labels and 12,500 optimizer updates per medium. TR-S saw 100,000 final-position
labels. The observed TR-C gains are therefore complete-configuration
differences, not estimates that history or memory caused the gain: readout
architecture, capacity, supervision density, and compute all changed. The
comparison is also one seed on heavily overlapping holdout contexts.

Verify the committed manifest, result files, per-example predictions, and any
locally present checkpoints from the `visual/` directory:

    python stlm_prototype/verify_tr_c_release.py

The historical run ID is read-only. A fresh independent replication from the
current source must use a new ID; the runner defaults to
`tr_c_replication_seed42_t128`:

    ./run_stlm_contextual_readouts.sh

The runner refuses a dirty worktree, authenticates the source SP, TR-S, and
direct-token checkpoints, trains or resumes both fresh TR-C configurations,
verifies that the source stayed unchanged, and collects all five results.
Changing the source, seed, window budget, or hyperparameters requires another
new run ID. Context length is fixed by the frozen SP checkpoint. For example:

    STLM_RUN_ID=tr_c_replication_seed43_t128 \
    STLM_SEED=43 \
    STLM_CONTEXTUAL_TRAIN_WINDOWS=100000 \
      ./run_stlm_contextual_readouts.sh

## Full SP + TR-S replication

From the repository root, start the first controlled experiment with:

    bash run_stlm_experiment.sh

The runner executes the text and pixel pipelines in order, writes logs to
`stlm_prototype/runs/replication_seed42_e1_n2-100k/`, and creates completion markers for
finished stages. Training checkpoints include completed-step and random-number
generator state. Ctrl-C and SIGTERM save the current stage; rerunning the same
command resumes it and skips completed stages.

The default is a matched baseline: seed 42, 50,000 TinyStories stories, one
**total** corpus epoch for each SP, and the same 100,000 sampled next-token
contexts for each TR-S. The readouts use the same fixed 1,024-context
holdout. Old checkpoints in `data/` are not reused. Each run also writes JSONL
metrics, text and pixel generation samples, and an H7 strip under `results/`.

To run the individual stages:

    python phase0_canonicals.py --native 512
    python phase0_text_canonicals.py
    python write_run_manifest.py
    python phase1_nn1.py
    python phase2_nn2.py
    python generate.py "how are you"
    python h7_demo.py "how are you"

Create the manifest after the fixed canonical artifacts exist and before any
training stage. Training now refuses to proceed without a matching manifest.

The corpus size and training budgets can be overridden for a smoke test or a
longer run without editing code:

    STLM_RUN_ID=paper_seed43_e1_n2-100k STLM_SEED=43 \
      bash run_stlm_experiment.sh

Use a new `STLM_RUN_ID` when changing the seed, corpus size, SP epoch targets,
or `STLM_READOUT_TRAIN_TARGETS`. The immutable `manifest.json` prevents accidental
reuse of a run ID with a different experiment configuration.

## Three phases

**Phase 0 -- visual canonicals.** For each word, Sana-Sprint renders a
512x512 RGB image from the LLM-authored symbolic description. The image is
also downsampled to 64x64 for training. Special tokens get solid placeholder
colors. Output: `data/canonicals.npz` of shape
`[VOCAB_SIZE, 3, 64, 64]` plus the native and downsampled PNGs for inspection.

**Phase 1.** Train the SP alone -- a causal transformer with a pixel-output
head. Target at each text position is the canonical pixel image of the
*next* token. Loss: per-pixel MSE.

**Phase 2.** Freeze the SP. For each sampled 128-token context, project only
the SP's final predicted substrate and train TR-S with cross-entropy to
classify it as the next token. Description and pixel TR-S receive the same
sampled context sequence and holdout positions. TR-S tunes to the SP's actual
output distribution,
so train-time and inference-time inputs match exactly.

## Files

- `../run_stlm_experiment.sh` -- resumable text-plus-pixel experiment runner.
- `../run_stlm_contextual_readouts.sh` -- complete five-condition comparison.
- `config.py` -- paths and hyperparameters.
- `training_utils.py` -- reproducibility and checkpoint helpers.
- `verify_published_results.py` -- read-only checkpoint/hash re-evaluation.
- `verify_tr_c_release.py` -- verify promoted TR-C artifacts and recompute
  metrics from all per-example predictions.
- `collect_tr_c_five_condition_results.py` -- consolidated comparison artifact.
- `models.py` -- SP, TR-S, and TR-C model definitions.
- `phase2_tr_c_pixel.py` -- train pixel TR-C.
- `phase2_tr_c_description.py` -- train description TR-C.
- `contextual_readout_training.py` -- shared resumable TR-C protocol.
- `vocab.py` -- the 1024-token tokenizer.
- `phase0_canonicals.py` -- the Sana-Sprint visual canonical generator.
- `phase1_nn1.py` -- train the pixel SP (legacy filename).
- `phase2_nn2.py` -- train pixel TR-S on frozen SP outputs (legacy filename).
- `generate.py` -- text-in, token-out loop.
- `generate_text.py` -- generation through the text-description substrate.
- `h7_demo.py` -- render the SP's emitted pixel images as a labelled strip.

## Data

- `data/word_symbolic_descriptions.json` -- per-word image descriptions used
  by Sana-Sprint.
- `data/512/<word>.png` -- native-resolution canonical images.
- `data/512/downsampling/<word>.png` -- 64x64 training images.
- `data/canonicals.npz` -- packed canonicals tensor.
- `runs/<run-id>/manifest.json` -- immutable experiment and runtime metadata.
- `runs/<run-id>/checkpoints/` -- isolated text and pixel checkpoints.
- `runs/<run-id>/logs/` -- complete stage logs and completion markers.
- `runs/<run-id>/results/` -- JSONL metrics, generation samples, and H7 strip.

The `runs/` tree is gitignored because checkpoints can be large. The reported
TR-C manifest, compact metrics, combined comparison, and per-example
predictions are force-tracked unchanged at their original run paths; their
hashes and byte sizes are recorded in
`published/tr_c_data5x_seed42_t128_release.json`. The exact eleven checkpoints
and canonical artifacts remain outside ordinary Git history and are hosted in
the companion [Hugging Face artifact repository](https://huggingface.co/emergent-wisdom/substrate-language-modeling).
Its `SHA256SUMS` manifest authenticates the downloaded files before they are
restored to the paths recorded in the release metadata.

## Reproducibility

External datasets, model snapshots, token streams, the vocabulary, and both
canonical tensors are pinned and hash-verified. New manifests record those
identities and the relevant package versions; a mismatch stops the run instead
of silently producing a different experiment. The legacy runs are recorded in
`published/pilot_runs.json`; the completed derived TR-C run has a separate
release record so the hash of that legacy registry remains historically stable.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact-evaluation,
artifact-restoration, and fresh-training procedures.
