# VEX Pixel Studio

Native Android pixel art editor with **real local 32×32 AI diffusion inference** (ONNX Runtime). English-only UI, square controls and sharp corners.

## Features implemented in source
- Manual pixel painting and erasing on 16×16, 32×32, 64×64 grids
- Palette picker and clear canvas
- Transparent PNG export using Android Storage Access Framework
- Category-conditioned 32×32 sprite generation: seed, category, 8/16/32 steps, actual ONNX runtime sampling, progress indicator
- Import a supported ONNX model from the phone. No network permission needed.
- GitHub Actions workflow building a debug APK artifact on pushes

## Important limitations
- Import a real **sprite-150k.onnx** model before using Generate. The model is not bundled.
- The Notes input does not control this category-only model; it is intentionally labeled.
- Text-conditioned **64×64 generation is not yet implemented**; the 64×64 canvas is manual editing.
- Project persistence, undo/redo and layer editing are not yet implemented.
- APK build and hardware inference are pending verification. Do not represent this code as device-tested.

Read [MODEL_SETUP.md](MODEL_SETUP.md) for verified model source and instructions.

## GitHub Actions
Open Actions > Android Debug APK > latest successful run > **VexPixelStudio-debug** artifact. The ZIP contains the debug APK.

## License
Model is maintained by its respective author; see Hugging Face model card for license. Model weights are not included in this repository.
