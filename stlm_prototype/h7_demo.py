"""
h7_demo.py -- display NN1's pixel outputs alongside the emitted tokens.

The H7 strip: one row of NN1's actual emitted pixel images, each labelled
with the token NN2 read it as. The images ARE the substrate -- no decoder
or paired generator needed.

    python h7_demo.py "how are you"
"""
import json
import sys

from PIL import Image, ImageDraw

import config
from generate import generate


def main():
    prompt = " ".join(sys.argv[1:]) or "how are you"
    tok, ids, images = generate(prompt)
    n_prompt = len(tok.encode(prompt, add_bos=True, add_eos=False))
    gen_ids = ids[n_prompt:]

    cell, pad, labelh = 96, 6, 16
    W = max(1, len(images)) * (cell + pad) + pad
    H = cell + labelh + 2 * pad
    strip = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(strip)
    for k, img in enumerate(images):
        arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        pic = Image.fromarray(arr).resize((cell, cell), Image.NEAREST)
        x = pad + k * (cell + pad)
        strip.paste(pic, (x, pad))
        word = tok.words[gen_ids[k]] if k < len(gen_ids) else "?"
        draw.text((x + 2, pad + cell + 2), word[:12], fill="black")

    out = config.RESULTS / "pixel_strip.png"
    strip.save(out)
    result = {
        "prompt": prompt,
        "output": tok.decode(ids),
        "run_id": config.RUN_ID,
    }
    result_out = config.RESULTS / "pixel_generation.json"
    result_out.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("prompt :", prompt)
    print("output :", result["output"])
    print("saved  :", out)
    print("saved  :", result_out)


if __name__ == "__main__":
    main()
