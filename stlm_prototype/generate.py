"""
generate.py -- the deployed loop: text -> NN1 pixel image -> NN2 -> token.

    python generate.py "how are you"

NN1 emits one 3x64x64 pixel image per position via a single forward pass.
NN2 (stateless CNN) names the image as a token. The token is appended;
NN1 reads on.
"""
import sys

import torch

import config
from models import ImageClassifier, ImagePredictor
from vocab import Tokenizer


def load_system():
    tok = Tokenizer()
    for ckpt in (config.NN1_CKPT, config.NN2_CKPT):
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{ckpt} missing -- run phases 0, 1, 2 first")

    n1 = torch.load(config.NN1_CKPT, map_location=config.DEVICE)
    if not n1.get("run_complete", False):
        raise RuntimeError("pixel NN1 training is not complete")
    nn1 = ImagePredictor(
        n_tokens=n1["n_tokens"],
        img_size=n1["img_size"], img_ch=n1["img_ch"],
        context_len=n1["context_len"], txt_layers=n1["txt_layers"],
        txt_d_model=n1["txt_d_model"], txt_heads=n1["txt_heads"])
    nn1.load_state_dict(n1["state_dict"])
    nn1 = nn1.to(config.DEVICE)
    nn1.train(False)

    n2 = torch.load(config.NN2_CKPT, map_location=config.DEVICE)
    if not n2.get("run_complete", False):
        raise RuntimeError("pixel NN2 training is not complete")
    nn2 = ImageClassifier(
        n2["img_size"], n2["n_tokens"],
        channels=tuple(n2["channels"]), fc_hidden=n2["fc_hidden"])
    nn2.load_state_dict(n2["state_dict"])
    nn2 = nn2.to(config.DEVICE)
    nn2.train(False)
    return tok, nn1, nn2


@torch.no_grad()
def generate(prompt, max_new=30):
    tok, nn1, nn2 = load_system()
    ids = tok.encode(prompt, add_bos=True, add_eos=False)
    images = []
    for _ in range(max_new):
        ctx = torch.tensor(
            [ids[-config.NN1_CONTEXT:]], device=config.DEVICE)
        img = nn1(ctx)[0, -1]              # [3, H, W] in [0, 1]
        tok_id = int(nn2(img[None])[0].argmax())
        images.append(img.cpu())
        ids.append(tok_id)
        if tok_id == tok.eos:
            break
    return tok, ids, images


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "how are you"
    tok, ids, _ = generate(prompt)
    print("prompt :", prompt)
    print("output :", tok.decode(ids))
