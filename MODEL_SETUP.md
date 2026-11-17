# Set up real offline image generation

The generator performs **actual on-device** 32×32 diffusion. It does not select pre-drawn images.

1. Open https://huggingface.co/Cyronius/sprite-diffusion/tree/main
2. Download **sprite-150k.onnx** (or sprite-100k.onnx) to your Android device.
3. Open VEX Pixel Studio > GENERATE > IMPORT SPRITE-DIFFUSION ONNX and select that file.
4. Choose one of the eight supported categories, a numeric seed, and 8/16/32 sampling steps.
5. Press GENERATE REAL SPRITE. The model executes with ONNX Runtime **on your phone**, with no server endpoint and no network permission.

This model only accepts categories: creature, item, weapon, food, effect, terrain, structure, nature. The Notes field is not passed to the model. Free-text 64×64 Sprite-GPT is **not implemented** yet: the original checkpoint is PyTorch and requires a verified Android pipeline.

The output is 32×32 RGBA. The 16×16/64×64 selectors only affect the **manual editor**. Generated pixel buffers are editable and can be exported to PNG through Android's Save dialog. The app does not yet persist projects or provide undo/redo.

If importing a nonmatching ONNX graph, installation fails without pretending a model is installed. Model data is kept in app-private storage and remains available offline.

## Technical specification

The inference loop follows the Hugging Face repository's model card: duplicate latent in batch two; category [c,8], footprint [3,3], CFG=3; Euler integration t=1→0 with N steps; transform [-1,1] output to 8-bit RGBA. ONNX model must expose x, t, category, footprint inputs and a velocity output.

## Licensing

Model weights are not committed to this repository. See the model's Apache 2.0 license and attribution requirements at the source.

## Caveats

No claim of measured inference speed or a successful tested APK until CI and physical-device test are complete.
