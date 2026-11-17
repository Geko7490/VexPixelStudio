# Install real on-device AI models

## Easiest path: in-app download

1. Open VEX Pixel Studio > **MODELS**.
2. Download **Sprite Diffusion 150k** (or **100k**) from Hugging Face.
3. See transferred bytes/progress. Tap CANCEL to interrupt; the `.part` file allows the next attempt to resume if the server supports HTTP Range.
4. The app opens the downloaded ONNX graph and validates the required inputs/output before accepting it.
5. Tap USE MODEL to switch between installed checkpoints.
6. Open GENERATE, select one of the 8 categories, choose 8/16/32 steps and a numeric seed.
7. Generate on-device without any further Internet access.

If you already have a compatible 32x32 ONNX file, use **IMPORT SPRITE-DIFFUSION ONNX** instead.

## 64x64

A 64×64 pixel editing canvas is included, but **64×64 AI generation isn't ready for users**.
Sprite-GPT upstream publishes PyTorch checkpoints, not Android-ready ONNX/CLIP. An experimental
conversion script and GitHub Actions workflow have been added:

- Tools: `tools/export_spritegpt64.py`
- Workflow: **Experimental 64x64 ONNX conversion**
- Modes: `class64`, `text64`

This workflow is a developer conversion/parity-check step only. Its ONNX output is not yet usable
through the Android generation UI. The text model additionally requires a local CLIP tokenizer and
encoder, and the 64×64 outputs are RGB over white rather than RGBA.

## Honest capability limits

32×32 Sprite Diffusion supports 8 category labels, **not open-ended text prompting**.
The Notes field is not sent to this model. It is not a hidden remote API or a template picker.
The weights are not bundled in the APK and aren't included in this Git repository.

Source: https://huggingface.co/Cyronius/sprite-diffusion (Apache-2.0).
64×64 source: https://huggingface.co/gmmeyer/sprite-gpt (CC-BY-SA-4.0).

### Testing

GitHub Actions must build successfully and a physical Android device must run actual inference
before this can be described as a verified release. The presence of these files alone isn't proof.
