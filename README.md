# VEX Pixel Studio

**Native Android pixel-art studio — English interface, sharp square corners.**

## v0.3.1 — Real offline prompt generation

Three distinct generation modes are implemented:

- **FAST 32** — Sprite Diffusion, an 8-category, 32×32 RGBA ONNX model (no natural-language prompt support).
- **PROMPT 32** — PtahX32, true 32×32 text-conditioned diffusion with CLIP, DDIM sampler and adjustable guidance.
- **PROMPT 64** — Sprite-GPT Text64, true 64×64 text-conditioned rectified-flow sampling with CLIP cross-attention.

**Every generated image comes from neural model inference on the phone. No static sprite bank is substituted.**
Both text-model generator weights and CLIP models were converted from upstream author checkpoints and passed PyTorch/ONNX parity checks in [this CI run](https://github.com/Geko7490/VexPixelStudio/actions/runs/44084764455). Their output is RGB on a light background, not real alpha; the manual editor and fast model support transparency.

### Install APK and models

1. Download **VexPixelStudio-debug** from the latest successful [Android Debug APK](https://github.com/Geko7490/VexPixelStudio/actions).
2. Open **MODELS**, then download PtahX32 (about 345 MB) or Sprite-GPT Text64 (about 420 MB). Both downloads are hosted as official assets in [models-v0.3](https://github.com/Geko7490/VexPixelStudio/releases/tag/models-v0.3).
3. Wait for the ONNX graph and SHA-256 validation to complete.
4. On **GENERATE**, choose **PROMPT 32** or **PROMPT 64**. Enter a short English description, set guidance and steps, and use genuine intermediate-sampler previews.
5. Export PNG or edit the generated pixels. Generation works entirely offline once the model is installed.

The model manager supports pause-by-cancel and partial-download resume when the server honors HTTP Range. Alternatively, import the single `ptah32-bundle.zip` or `spritegpt64-bundle.zip` file from your phone. **Do not import the outer GitHub Actions artifact ZIP**; extract its bundle ZIP first.

### Development tools

- `tools/export_text_bundles.py`: exports both real text-model generators and CLIP text encoders as ONNX, verifies ONNX/PyTorch parity, writes SHA-256 manifest and ZIP bundle
- `.github/workflows/build-text-models.yml`: produces validated model artifacts
- `.github/workflows/publish-models.yml`: publishes the verified bundles
- `TextSpriteEngine.kt`: local Android DDIM and Euler CFG samplers
- `ClipBpeTokenizer.kt`: offline CLIP byte-level BPE tokenizer
- `TextBundleStore.kt`: secure model package downloader/installer

### Limitations

- **Real Android inference performance, visual quality and memory use require physical-device testing.** A successful APK build and ONNX parity on desktop do not certify successful phone generation.
- PtahX32 and Sprite-GPT64 produce **RGB**, not native transparent backgrounds; cleanup/transparency tooling is not yet complete.
- Large downloads and inference can consume storage, RAM, battery and time.
- Save-project history, multilayer editor, undo/redo and image-to-animation export are not yet complete.

### Attribution

- Sprite Diffusion: https://huggingface.co/Cyronius/sprite-diffusion — Apache-2.0
- PtahX32: https://huggingface.co/viknesh100/PtahX32 — MIT checkpoint, review training data terms separately
- Sprite-GPT: https://huggingface.co/gmmeyer/sprite-gpt — CC-BY-SA-4.0
- CLIP text encoder: https://huggingface.co/openai/clip-vit-base-patch32 — MIT

These are third-party models. See the upstream repositories for licensing and attribution.
