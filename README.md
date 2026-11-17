# VEX Pixel Studio

Native Android pixel art studio. **English-only UI, sharp square corners, offline-first.**

## Current state
- Kotlin + Jetpack Compose Android shell
- Generate / Editor / Projects navigation
- 16×16, 32×32, 64×64 transparent pixel grid preview
- Palette swatches and prompt input
- AI generation deliberately disabled until a real model is integrated

**Important:** This commit does not contain a trained AI model, ONNX inference, functioning pixel drawing gestures, export, or persistence. No mock AI generations are shown.

## Next development steps
1. Verify real 64×64 model weights, license and inference procedure.
2. Convert to ONNX or choose a verified Android-compatible inference backend.
3. Implement actual sampler, model installation and progress reporting.
4. Implement touch drawing, undo/redo, PNG export and project persistence.
5. Add GitHub Actions Android APK build and test on a physical device.

## Build
Use JDK 17 and Android SDK 35, then run `gradle :app:assembleDebug` with a compatible Gradle installation.
