#!/usr/bin/env python3
"""Export the upstream 64x64 Sprite-GPT velocity network to ONNX.

This is an *offline conversion utility*, not an Android runtime. The text
checkpoint additionally needs a local CLIP tokenizer + encoder for prompts.

Prerequisites:
  git clone https://github.com/gmmeyer/sprite-diffusion /tmp/sprite-diffusion
  PYTHONPATH=/tmp/sprite-diffusion python tools/export_spritegpt64.py --mode class64

Outputs are saved under dist/ and MUST be checked for parity before release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from spritegpt.unet import UNet


class ClassVelocity(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x, t, y):
        return self.network(x, t, y)


class TextVelocity(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x, t, ctx, ctx_mask):
        return self.network(x, t, ctx=ctx, ctx_mask=ctx_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["class64", "text64"], required=True)
    ap.add_argument("--output-dir", default="dist")
    args = ap.parse_args()

    checkpoint_name = "sprite-gpt-cond64.pt" if args.mode == "class64" else "sprite-gpt-text64.pt"
    checkpoint = hf_hub_download("gmmeyer/sprite-gpt", filename=checkpoint_name)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = UNet(
        img_size=cfg["img_size"],
        base=cfg["base"],
        ch_mult=tuple(map(int, cfg["ch_mult"].split(","))),
        num_res=cfg["num_res"],
        attn_res=tuple(map(int, cfg["attn_res"].split(","))),
        num_classes=cfg["num_classes"],
        ctx_dim=cfg.get("ctx_dim"),
        dropout=0.0,
    )
    net.load_state_dict(ck["ema_state_dict"])
    net.eval()

    x = torch.zeros(2, 3, 64, 64, dtype=torch.float32)
    t = torch.ones(2, dtype=torch.float32)
    if args.mode == "class64":
        assert cfg["num_classes"] == 31, "Unexpected class model configuration"
        module = ClassVelocity(net).eval()
        tensors = (x, t, torch.tensor([0, 31], dtype=torch.int64))
        input_names = ["x", "t", "y"]
    else:
        assert cfg.get("ctx_dim") == 512, "Unexpected text embedding dimension"
        module = TextVelocity(net).eval()
        tensors = (
            x,
            t,
            torch.zeros(2, 32, 512, dtype=torch.float32),
            torch.ones(2, 32, dtype=torch.bool),
        )
        input_names = ["x", "t", "ctx", "ctx_mask"]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"sprite-gpt-{args.mode}.onnx"
    with torch.inference_mode():
        expected = module(*tensors).numpy()
        torch.onnx.export(
            module,
            tensors,
            str(target),
            input_names=input_names,
            output_names=["velocity"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    result = session.run(["velocity"], dict(zip(input_names, [v.numpy() for v in tensors])))[0]
    assert result.shape == (2, 3, 64, 64), result.shape
    max_error = float(np.max(np.abs(expected - result)))
    assert max_error < 0.003, f"ONNX parity failed: max error={max_error}"
    (out / f"sprite-gpt-{args.mode}.json").write_text(json.dumps({
        "source": "https://huggingface.co/gmmeyer/sprite-gpt",
        "checkpoint": checkpoint_name,
        "inputs": input_names,
        "max_abs_error": max_error,
        "output": "velocity",
        "license": "CC-BY-SA-4.0",
        "note": "This export alone does not implement Android inference or CLIP tokenization.",
    }, indent=2))
    print(f"Export verified: {target}, max abs error={max_error:.6g}")


if __name__ == "__main__":
    main()
