"""
vocab.py -- the toy's word-level vocabulary and tokenizer.

The vocabulary is the most frequent words in the training corpus itself
(TinyStories), plus three special tokens. Building it from the corpus
guarantees every vocabulary word actually occurs in phase-two training data.

    python vocab.py        # (re)builds data/vocab.json
"""
import json
import re
from collections import Counter

import config

SPECIALS = ["<bos>", "<eos>", "<unk>"]
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")   # words; keeps contractions


def tokenize(text):
    """Lowercase and extract word tokens. Word-level: no subwords."""
    return _WORD_RE.findall(text.lower())


def build_vocab(scan=None):
    """Count words across the corpus, keep the most common, save vocab.json."""
    from datasets import load_dataset
    scan = scan or config.MAX_SENTENCES * 4
    print(f"loading {config.CORPUS_NAME}@{config.CORPUS_REVISION[:12]} ...")
    ds = load_dataset(
        config.CORPUS_NAME,
        revision=config.CORPUS_REVISION,
        split="train",
    )
    if getattr(ds, "_fingerprint", None) != config.CORPUS_DATASET_FINGERPRINT:
        raise RuntimeError("TinyStories fingerprint mismatch")
    counter = Counter()
    for i, ex in enumerate(ds):
        if i >= scan:
            break
        counter.update(tokenize(ex["text"]))
    n_content = config.VOCAB_SIZE - len(SPECIALS)
    common = [w for w, _ in counter.most_common(n_content)]
    words = SPECIALS + common
    config.VOCAB_FILE.write_text(json.dumps(words))
    print(f"vocab: {len(words)} tokens "
          f"({len(SPECIALS)} special + {len(common)} words) "
          f"-> {config.VOCAB_FILE}")
    return words


class Tokenizer:
    """Word-level tokenizer over the saved vocabulary."""

    def __init__(self):
        if not config.VOCAB_FILE.exists():
            raise FileNotFoundError(
                f"{config.VOCAB_FILE} missing -- run `python vocab.py` first")
        self.words = json.loads(config.VOCAB_FILE.read_text())
        self.word2id = {w: i for i, w in enumerate(self.words)}
        self.bos = self.word2id["<bos>"]
        self.eos = self.word2id["<eos>"]
        self.unk = self.word2id["<unk>"]

    def __len__(self):
        return len(self.words)

    def encode(self, text, add_bos=True, add_eos=False):
        ids = [self.word2id.get(w, self.unk) for w in tokenize(text)]
        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]
        return ids

    def decode(self, ids):
        return " ".join(self.words[i] for i in ids
                        if self.words[i] not in SPECIALS)

    def in_vocab(self, text):
        """True if `text` is non-empty and every word is in the vocabulary."""
        words = tokenize(text)
        return bool(words) and all(w in self.word2id for w in words)


if __name__ == "__main__":
    build_vocab()
