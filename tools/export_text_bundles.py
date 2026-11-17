#!/usr/bin/env python3
"""Convert real upstream text-to-pixel-art checkpoints into verified Android ONNX ZIP bundles.

Usage:
    python tools/export_text_bundles.py --mode ptah32
    python tools/export_text_bundles.py --mode spritegpt64 --upstream /tmp/sprite-diffusion

This script generates *real* model artifacts from the author's published weights.
Bundle contains generator.onnx, clip.onnx, vocab.json, merges.txt, manifest.json.
No placeholder / synthesized output, no server inference.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from transformers import CLIPTextModel, CLIPTokenizer


CLIP_REPO = "openai/clip-vit-base-patch32"


class ClipModel(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input_ids, attention_mask):
        return self.net(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class PtahVelocity(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t, ctx):
        return self.net(x, t, encoder_hidden_states=ctx).sample


class SpriteGptVelocity(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t, ctx, ctx_mask):
        return self.net(x, t, ctx=ctx, ctx_mask=ctx_mask.to(torch.bool))


def export_and_compare(net, tensors, target, inputs, output, tolerance):
    net.eval()
    with torch.inference_mode():
        expected = net(*tensors).float().cpu().numpy()
        torch.onnx.export(
            net, tensors, str(target), input_names=inputs, output_names=[output],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    inputs_map = dict(zip(inputs, [x.detach().numpy() for x in tensors]))
    with torch.no_grad():
        actual = session.run([output], inputs_map)[0]
    assert expected.shape == actual.shape, f"Export shape mismatch {expected.shape} vs {actual.shape}"
    err = float(np.max(np.abs(expected - actual)))
    assert np.isfinite(actual).all() and err < tolerance, f"ONNX/PyTorch parity failed, max error {err}"
    return err


def generate(args):
    torch.set_num_threads(4)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    is32 = args.mode == "ptah32"
    max_tokens = 24 if is32 else 32
    res = 32 if is32 else 64
    directory = root / args.mode
    directory.mkdir(exist_ok=True)
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_REPO)
    tokenizer.save_pretrained(directory)
    text_model = CLIPTextModel.from_pretrained(CLIP_REPO).cpu().eval()

    sample = tokenizer(
        ["", "a blue bird sprite"], padding="max_length",
        max_length=max_tokens, truncation=True, return_tensors="pt",
    )
    clip_inputs = (sample.input_ids.cpu(), sample.attention_mask.cpu())
    clip_net = ClipModel(text_model)
    clip_error = export_and_compare(
        clip_net, clip_inputs, directory / "clip.onnx",
        ["input_ids", "attention_mask"], "last_hidden_state", tolerance=0.003,
    )
    del text_model, clip_net

    x = torch.randn(2, 3, res, res)
    if is32:
        from diffusers import UNet2DConditionModel
        checkpoint = hf_hub_download("viknesh100/PtahX32", "checkpoints/best.pt")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = UNet2DConditionModel(**ck["arch"])
        model.load_state_dict(ck["ema"])
        net = PtahVelocity(model.cpu().eval())
        model_inputs = (
            x, torch.tensor([980, 980], dtype=torch.long),
            torch.randn(2, max_tokens, 512),
        )
        input_names = ["x", "t", "ctx"]
        sampler = "ddim"
        source = "https://huggingface.co/viknesh100/PtahX32"
        license_name = "MIT (review source dataset/third-party terms)"
    else:
        sys.path.insert(0, str(Path(args.upstream).resolve()))
        from spritegpt.unet import UNet
        checkpoint = hf_hub_download("gmmeyer/sprite-gpt", "sprite-gpt-text64.pt")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        model = UNet(
            img_size=cfg["img_size"], base=cfg["base"],
            ch_mult=tuple(map(int, cfg["ch_mult"].split(","))),
            num_res=cfg["num_res"],
            attn_res=tuple(map(int, cfg["attn_res"].split(","))),
            dropout=0.0, num_classes=cfg["num_classes"], ctx_dim=cfg.get("ctx_dim"),
        )
        model.load_state_dict(ck["ema_state_dict"])
        net = SpriteGptVelocity(model.cpu().eval())
        model_inputs = (
            x, torch.tensor([0.8, 0.8], dtype=torch.float32),
            torch.randn(2, max_tokens, 512),
            torch.ones(2, max_tokens, dtype=torch.int64),
        )
        input_names = ["x", "t", "ctx", "ctx_mask"]
        sampler = "euler"
        source = "https://huggingface.co/gmmeyer/sprite-gpt"
        license_name = "CC-BY-SA-4.0"

    generator_error = export_and_compare(
        net, model_inputs, directory / "generator.onnx",
        input_names, "velocity", tolerance=0.02,
    )
    manifest = {
        "format_version": 1,
        "model_id": args.mode,
        "resolution": res,
        "sampler": sampler,
        "max_tokens": max_tokens,
        "source": source,
        "license": license_name,
        "offline": True,
        "clip_source": CLIP_REPO,
        "clip_parity_max_error": clip_error,
        "generator_parity_max_error": generator_error,
        "sha256": {},
    }
    for name in ["generator.onnx", "clip.onnx", "vocab.json", "merges.txt"]:
        manifest["sha256"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    path = root / (args.mode + "-bundle.zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in ["generator.onnx", "clip.onnx", "vocab.json", "merges.txt", "manifest.json"]:
            archive.write(directory / name, arcname=name)
    print("Verified ONNX bundle:", path, "bytes=", path.stat().st_size)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ptah32", "spritegpt64"), required=True)
    parser.add_argument("--output-dir", default="dist-text-models")
    parser.add_argument("--upstream", default="/tmp/sprite-diffusion")
    generate(parser.parse_args())
