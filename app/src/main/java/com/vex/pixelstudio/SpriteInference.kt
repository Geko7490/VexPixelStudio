package com.vex.pixelstudio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.io.File
import java.util.Random
import kotlin.math.roundToInt

/**
 * Actual category-conditioned Sprite Diffusion ONNX inference, not templates.
 * Source: https://huggingface.co/Cyronius/sprite-diffusion
 * The checkpoint has a fixed [2,4,32,32] input and returns velocity.
 */
class SpriteInference(private val context: Context) {
    private val modelFile: File get() = File(context.filesDir, "sprite-diffusion.onnx")

    fun installed(): Boolean = modelFile.exists() && modelFile.length() > 1_000_000

    fun importModel(uri: Uri) {
        val tmp = File(context.filesDir, "sprite-diffusion.tmp")
        try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                tmp.outputStream().use { output -> input.copyTo(output) }
            } ?: error("Unable to open model file")
            require(tmp.length() > 1_000_000) { "Model file is too small" }
            val env = OrtEnvironment.getEnvironment()
            env.createSession(tmp.absolutePath, OrtSession.SessionOptions()).use { session ->
                val names = session.inputNames
                require(names.containsAll(listOf("x", "t", "category", "footprint"))) { "Wrong model: required Sprite Diffusion inputs not found" }
            }
            require(tmp.renameTo(modelFile)) { "Could not install model" }
        } finally { if (tmp.exists()) tmp.delete() }
    }

    fun generate(category: Int, seed: Long, steps: Int, progress: (Int) -> Unit): Bitmap {
        require(installed()) { "Import an ONNX model first" }
        require(category in 0..7)
        require(steps in listOf(8, 16, 32))
        val env = OrtEnvironment.getEnvironment()
        val random = Random(seed)
        val latent = FloatArray(2 * 4 * 32 * 32)
        for (i in 0 until 4 * 32 * 32) {
            val value = random.nextGaussian().toFloat()
            latent[i] = value
            latent[i + 4 * 32 * 32] = value
        }
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(4)
        env.createSession(modelFile.absolutePath, options).use { session ->
            for (step in 0 until steps) {
                val t = 1f - step.toFloat() / steps
                val x = OnnxTensor.createTensor(env, FloatBuffer.wrap(latent), longArrayOf(2, 4, 32, 32))
                val time = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(t, t)), longArrayOf(2))
                val cls = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(category.toLong(), 8L)), longArrayOf(2))
                val footprint = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(3L, 3L)), longArrayOf(2))
                try {
                    session.run(mapOf("x" to x, "t" to time, "category" to cls, "footprint" to footprint)).use { result ->
                        val velocity = (result[0].value as Array<*>)
                        @Suppress("UNCHECKED_CAST")
                        val rows = velocity as Array<Array<Array<FloatArray>>>
                        for (channel in 0 until 4) for (y in 0 until 32) for (xx in 0 until 32) {
                            val i = channel * 1024 + y * 32 + xx
                            val cond = rows[0][channel][y][xx]
                            val uncond = rows[1][channel][y][xx]
                            val guided = uncond + 3f * (cond - uncond)
                            latent[i] -= guided / steps
                            latent[i + 4096] = latent[i]
                        }
                    }
                } finally { x.close(); time.close(); cls.close(); footprint.close() }
                progress(((step + 1) * 100) / steps)
            }
        }
        val pixels = IntArray(1024)
        for (i in pixels.indices) {
            fun byte(channel: Int): Int = (((latent[channel * 1024 + i].coerceIn(-1f, 1f) + 1f) * 127.5f).roundToInt()).coerceIn(0, 255)
            val alpha = byte(3)
            pixels[i] = ((if (alpha >= 128) alpha else 0) shl 24) or (byte(0) shl 16) or (byte(1) shl 8) or byte(2)
        }
        return Bitmap.createBitmap(pixels, 32, 32, Bitmap.Config.ARGB_8888)
    }
}
