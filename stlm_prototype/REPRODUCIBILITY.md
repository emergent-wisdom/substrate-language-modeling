# Reproducing the pilot

The reported runs do not need to be repeated to repair a measurement. The
three legacy checkpoints re-evaluate to their recorded metrics, while the
promoted TR-C predictions reconstruct the new top-k counts exactly and NLL
within the stated serialization tolerance. The reproducibility work here
prevents external inputs from silently changing and packages the evidence
needed to verify those runs elsewhere.

## Frozen inputs

- TinyStories is loaded from the immutable Hugging Face revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, not from the moving `main`
  branch.
- The 50,000- and 250,000-story token streams are checked against their exact
  token counts and SHA-256 digests.
- The vocabulary and both canonical tensors are checked by SHA-256 before a
  manifest is created. The tensors are published in the companion Hugging Face
  artifact repository rather than ordinary Git history.
- MiniLM and Sana are loaded from immutable model revisions when phase zero is
  regenerated.
- `../requirements-repro.txt` records the complete Python environment used for
  verification.
- Every new run manifest records the corpus revision, dataset fingerprint,
  token-stream digest, canonical hashes, package versions, platform, device,
  and Git state.
- New manifests require a clean worktree and refuse to resume under a
  different commit, package environment, platform, or device.
- All four reported run IDs are read-only. Independent replications use a new
  run ID so they do not overwrite historical checkpoints or metrics.

The original two manifests predate the new corpus fields and have not been
rewritten. Their exact token streams were recovered from the cached immutable
revision and are recorded separately in `published/pilot_runs.json`. The
loader recognizes only those two run IDs as legacy and fails closed if their
story count, token count, or digest differs.

## Three different verification and reproduction claims

**Recorded-result verification** means authenticating the promoted TR-C
manifest, metrics, combined result, and per-example predictions and
recomputing NLL and top-k accuracy from those predictions. It verifies the two
TR-C rows in the paper without requiring a model checkpoint:

```sh
python stlm_prototype/verify_tr_c_release.py
```

**Exact evaluation reproduction** means loading the archived checkpoints and
recovering the published metrics on the fixed holdout. This is the appropriate
check that the reported numbers were transcribed and evaluated correctly.
`published/pilot_runs.json` contains hashes for the three legacy run IDs;
`published/tr_c_data5x_seed42_t128_release.json` records the two TR-C checkpoint
hashes and the immutable derived-run artifacts. The legacy verifier reloads and
evaluates its checkpoints. The compact TR-C verifier authenticates the two
checkpoint files when present, but reconstructs the TR-C metrics from the
promoted per-example predictions rather than reloading the models; the two
operations should not be conflated.

**Independent training replication** means training a fresh model under the
same recorded inputs and protocol. Seeds and RNG streams are fixed, but Apple
MPS does not promise bit-for-bit identical floating-point training across
hardware and software releases. A replication should therefore use a new run
ID, repeat multiple seeds, and compare distributions or declared numerical
tolerances rather than require identical checkpoint bytes.

## Checkpoint and canonical archive

The companion [Hugging Face artifact repository](https://huggingface.co/emergent-wisdom/substrate-language-modeling)
hosts the checkpoint sets for all four reported run IDs together with their
manifests, metrics, canonical artifacts, and a complete `SHA256SUMS` file. The
eleven checkpoints total 955,969,421 bytes (956.0 MB, or about 911.7 MiB) and
therefore remain outside ordinary Git history. Verify every downloaded file
against `SHA256SUMS` and the two published release records before evaluation.

The artifact repository enables exact-checkpoint evaluation in addition to
fresh training from frozen inputs. A DOI-bearing snapshot can mirror the same
authenticated files without changing their recorded paths or hashes.

After placing an authenticated checkpoint archive under the corresponding
`runs/<run-id>/checkpoints/` directory, re-evaluate the legacy checkpoints and
authenticate the TR-C checkpoint bytes without modifying them:

```sh
python stlm_prototype/verify_published_results.py data5x_seed42
python stlm_prototype/verify_published_results.py paper_seed42_e1_n2-100k
python stlm_prototype/verify_published_results.py baseline5x_seed42
python stlm_prototype/verify_tr_c_release.py --require-checkpoints
```

The legacy verifier reconstructs the pinned token stream, evaluates the
readouts or direct-token model on the fixed holdout, and compares full-precision
metrics with `published/pilot_runs.json`. The TR-C verifier authenticates both
derived checkpoint files when present and, separately, recomputes top-k counts
exactly and NLL within `1e-7` from all 1,024 promoted predictions; the tolerance
accounts for serializing per-example log probabilities from float32. It does
not claim that checkpoint hashing is model re-evaluation.

## Completed five-condition causally masked sequence-readout comparison

The TR-C configuration is derived from the published `data5x_seed42`
run. It does not retrain or copy the two substrate predictors (SPs), the two
single-state token readouts (TR-S), or the direct-token baseline. Instead, it
freezes each SP and freshly initializes one causal token readout (TR-C) per
medium. The completed TR-S and direct-token checkpoints are loaded only to
re-evaluate the three existing comparison conditions.

The reported run `tr_c_data5x_seed42_t128` completed with the authenticated
inputs and configuration recorded in its immutable manifest. Its results are:

| Configuration | NLL | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|
| Direct-token LM | 2.208 | 44.7% | 74.6% | 85.3% |
| Description SP + TR-C | 2.616 | 41.9% | 68.6% | 78.0% |
| Pixel SP + TR-C | 2.954 | 37.8% | 61.6% | 72.2% |
| Description SP + TR-S | 3.541 | 35.0% | 54.2% | 62.8% |
| Pixel SP + TR-S | 4.283 | 27.8% | 44.3% | 52.9% |

The historical run ID is read-only. From later code, a fresh replication uses
a new ID; the runner defaults to `tr_c_replication_seed42_t128`:

```sh
./run_stlm_contextual_readouts.sh
```

Run this command from `visual/` in a clean worktree. The manifest binds every
replication to its own exact code revision. The reported configuration was:

- derived run: `tr_c_data5x_seed42_t128`;
- frozen source run: `data5x_seed42`;
- context length: 128 frozen-SP states, including the current state;
- seed: 42;
- sampled training windows: 100,000, identical across media;
- supervised positions: 12.8 million per TR-C, from same-index
  cross-entropy at all 128 aligned positions;
- batch size: 8;
- readout: a fresh medium-specific state adapter shared across positions,
  mapping to 256 dimensions, followed by a two-layer, eight-head causal
  Transformer and per-position vocabulary logits;
- evaluation targets: the final position of the existing fixed 1,024-context
  holdout.

TR-C receives only the complete causal sequence of frozen-SP substrate outputs.
Raw token IDs and SP hidden states are not readout inputs.

The runner performs the full protocol in one invocation:

1. create or validate the immutable derived-run manifest;
2. authenticate both source SP checkpoints used by TR-C and the two TR-S and
   direct-token checkpoints used as comparison conditions, using
   `published/pilot_runs.json`;
3. re-evaluate the published pixel TR-S, description TR-S, and direct-token
   conditions;
4. train or resume description TR-C and pixel TR-C from fresh initializations;
5. authenticate both frozen source SP checkpoints again;
6. write all five conditions below the new replication run ID.

Ctrl-C or SIGTERM saves the active TR-C checkpoint. Running the same command
again validates the manifest and source, resumes incomplete training, and
rechecks every condition. Derived checkpoints are written only beneath the new
run directory. The source-run paths are never write targets.

Source run, seed, window budget, and model hyperparameters are configurable,
but every changed configuration needs a new derived run ID. Context length is
fixed by the frozen source SP checkpoint. For example:

```sh
STLM_SOURCE_RUN_ID=data5x_seed42 \
STLM_CONTEXTUAL_TRAIN_WINDOWS=100000 \
STLM_RUN_ID=tr_c_replication_seed43_t128 \
STLM_SEED=43 \
  ./run_stlm_contextual_readouts.sh
```

Only source runs recorded in `published/pilot_runs.json` are accepted. The
derived corpus token count and SHA-256 digest must match the selected source,
so a source override cannot silently change the training stream.

This five-row comparison is intentionally reported as a configuration
comparison, not as a controlled estimate of history or memory. TR-C differs
from TR-S in architecture and capacity and receives a label at every position:
12.8 million supervised positions versus the completed TR-S run's 100,000
final-position labels. Description TR-C records a 6.9-point top-1 gain over its
TR-S condition and pixel TR-C a 10.0-point gain, but those differences do not
identify their cause. The completed TR-S and direct-token measurements are
unchanged.
