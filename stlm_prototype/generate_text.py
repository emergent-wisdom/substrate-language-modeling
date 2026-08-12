"""Generate through the text-description substrate and save a sample result."""
import json
import sys

import torch

import config
from models import DescriptionEmbeddingClassifier, DescriptionEmbeddingPredictor
from vocab import Tokenizer


def load_system():
    tok = Tokenizer()
    for checkpoint in (config.NN1_TEXT_CKPT, config.NN2_TEXT_CKPT):
        if not checkpoint.exists():
            raise FileNotFoundError(f"{checkpoint} missing -- run text phases first")

    n1 = torch.load(config.NN1_TEXT_CKPT, map_location=config.DEVICE)
    n2 = torch.load(config.NN2_TEXT_CKPT, map_location=config.DEVICE)
    if not n1.get("run_complete") or not n2.get("run_complete"):
        raise RuntimeError("text-substrate training is not complete")

    nn1 = DescriptionEmbeddingPredictor(
        n_tokens=n1["n_tokens"], desc_len=n1["desc_len"],
        embed_dim=n1["embed_dim"], context_len=n1["context_len"],
        txt_layers=n1["txt_layers"], txt_d_model=n1["txt_d_model"],
        txt_heads=n1["txt_heads"])
    nn1.load_state_dict(n1["state_dict"])
    nn1 = nn1.to(config.DEVICE)
    nn1.train(False)

    nn2 = DescriptionEmbeddingClassifier(
        embed_dim=n2["embed_dim"], n_tokens=n2["n_tokens"],
        desc_len=n2["desc_len"], d_model=n2["d_model"],
        n_heads=n2["n_heads"], n_layers=n2["n_layers"],
        ff_dim=n2["ff_dim"])
    nn2.load_state_dict(n2["state_dict"])
    nn2 = nn2.to(config.DEVICE)
    nn2.train(False)
    return tok, nn1, nn2


@torch.no_grad()
def generate(prompt, max_new=30):
    tok, nn1, nn2 = load_system()
    ids = tok.encode(prompt, add_bos=True, add_eos=False)
    for _ in range(max_new):
        context = torch.tensor(
            [ids[-config.NN1_CONTEXT:]], device=config.DEVICE)
        substrate = nn1(context)[0, -1]
        token_id = int(nn2(substrate[None])[0].argmax())
        ids.append(token_id)
        if token_id == tok.eos:
            break
    return tok, ids


def main():
    prompt = " ".join(sys.argv[1:]) or "the dog ran into the yard"
    tok, ids = generate(prompt)
    output = tok.decode(ids)
    result = {"prompt": prompt, "output": output, "run_id": config.RUN_ID}
    out = config.RESULTS / "text_generation.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("prompt :", prompt)
    print("output :", output)
    print("saved  :", out)


if __name__ == "__main__":
    main()
