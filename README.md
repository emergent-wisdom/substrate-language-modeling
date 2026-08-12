# Substrate Language Modeling (STLM)

STLM is a research program for language models that must route every lexical
prediction through a structured non-token representation. Its long-term goal is
scene-mediated language generation: a model that builds a persistent physical
and schematic scene as it writes and uses that scene as the compulsory
interface to every next word.

This repository contains the paper and its reproducible stage-1 pilot. The
pilot tests the prerequisite interface with fixed pixel and description
targets. It shows that both interfaces can carry useful next-token signal, but
it does **not** establish that target meaning is causally used or that a
persistent scene has emerged. The first causal alignment experiment is the
paper's aligned-versus-permuted target control; the broader scene-emergence
hypothesis requires later-stage persistent or context-dependent substrates and
structure-sensitive tasks.

## Paper

- [`substratelm.pdf`](substratelm.pdf) - compiled preprint
- [`substratelm.tex`](substratelm.tex) - paper source

## Code and evidence

- [`stlm_prototype/README.md`](stlm_prototype/README.md) - architecture,
  results, and training commands
- [`stlm_prototype/REPRODUCIBILITY.md`](stlm_prototype/REPRODUCIBILITY.md) -
  recorded-result verification, exact checkpoint evaluation, and independent
  replication
- [`stlm_prototype/published/`](stlm_prototype/published/) - immutable release
  records for the reported runs
- [`stlm_prototype/runs/tr_c_data5x_seed42_t128/`](stlm_prototype/runs/tr_c_data5x_seed42_t128/) -
  promoted TR-C manifest, compact metrics, and per-example predictions

Verify the compact TR-C evidence without downloading model checkpoints:

```sh
python stlm_prototype/verify_tr_c_release.py
```

Run the causal-readout replication protocol after restoring the authenticated
artifacts:

```sh
./run_stlm_contextual_readouts.sh
```

## Large artifacts

The exact eleven reported checkpoints, canonical tensors, and canonical images
are hosted separately in the
[Hugging Face artifact repository](https://huggingface.co/emergent-wisdom/substrate-language-modeling).
That repository preserves the paths expected by the code and provides a
`SHA256SUMS` manifest. These files are intentionally excluded from ordinary Git
history.

The paper and its self-contained source package are released separately through
Zenodo. The GitHub repository is the code snapshot; Hugging Face is the large
artifact archive; Zenodo is the archival paper record.

## Reported result boundary

On the fixed 1,024-context TinyStories holdout, the direct-token baseline reaches
44.7% top-1 accuracy. Description and pixel causal readouts reach 41.9% and
37.8%; their stateless counterparts reach 35.0% and 27.8%. The causal and
stateless readouts are not capacity-, supervision-, or compute-matched, so these
figures compare complete configurations rather than isolate a memory effect.
All reported trained conditions use one seed.

## License

Source code is available under the [MIT License](LICENSE). The paper,
documentation, figures, vocabulary metadata, published results, and released
research artifacts are available under
[Creative Commons Attribution 4.0](LICENSE-CONTENT), except where third-party
terms apply.
