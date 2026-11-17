# VEX Pixel Studio

Native, English-language Android pixel art editor with **sharp square controls**.

## Added on 2026-11-17

- In-app **Models** screen with direct HTTPS downloads from the published Hugging Face repository
- Select/download/delete **Sprite Diffusion 150k** and **Sprite Diffusion 100k** ONNX checkpoints
- Byte-based download progress, cancel, and resumable `.part` file handling
- ONNX graph validation before a download becomes an installed model
- Select active 32×32 model without reinstalling the app
- Generate genuine 32×32 RGBA sprites via local ONNX Runtime; no generated sprite templates
- English UI, sharp-edged controls; manual 16×16/32×32/64×64 pixel editor
- PNG export through Android's file picker, and manual ONNX import
- GitHub Actions debug APK build on push

### App model setup

1. Install debug APK from the first **successful** Android Debug APK GitHub Actions run.
2. Tap **MODELS**, then **DOWNLOAD** beside Sprite Diffusion 150k (or 100k).
3. Let the app validate and install the model. Downloaded weights are held in app-private storage.
4. On **GENERATE**, choose a category and seed, then press **GENERATE REAL SPRITE**.
5. After installation, generation does not use the network.

The app requires Internet **only for model downloads**. Downloading model files from Hugging Face may require several hundred megabytes of traffic.

### Important 64×64 limitation

**64×64 manual editing is present**, but genuine 64×64 AI generation is **not yet functional**. The 64×64 Sprite-GPT options are listed as unavailable to avoid implying otherwise. Upstream publishes class-conditioned and CLIP text-conditioned PyTorch `.pt` checkpoints, not ready-to-use Android ONNX inference packages. These models cannot be passed to the existing 32×32 ONNX sampler unchanged. We must export/validate a compatible generator and (for text) CLIP tokenizer/encoder before making a working download button.

32×32 Sprite Diffusion accepts category labels rather than natural-language prompt control. The Notes field is not used by the model.

### Quality and test status

Source commits have been pushed through the connected GitHub app. **No confirmed successful Android build or physical-device inference test is claimed in this README.** Test an APK before using it as a production release.

### Sources & licensing

- https://huggingface.co/Cyronius/sprite-diffusion — Apache-2.0 model.
- https://huggingface.co/gmmeyer/sprite-gpt — CC-BY-SA-4.0. 64×64 reference model only at present.

Neither model checkpoint is committed to this Git repository.
