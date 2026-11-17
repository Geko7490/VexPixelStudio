package com.vex.pixelstudio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Random
import kotlin.coroutines.coroutineContext
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Real offline prompt-to-sprite inference using verified ONNX bundles.
 *
 * ptah32: DDIM epsilon prediction + CFG at 32x32
 * spritegpt64: rectified-flow velocity + Euler sampling + CFG at 64x64
 */
class TextSpriteEngine(private val context: Context) {
    private val store = TextBundleStore(context)

    suspend fun generate(
        modelId: String,
        prompt: String,
        seed: Long,
        steps: Int,
        guidance: Float,
        progress: suspend (Int) -> Unit
    ): Bitmap = withContext(Dispatchers.Default) {
        require(store.isInstalled(modelId)) { "Import a verified ONNX model bundle first" }
        require(prompt.isNotBlank()) { "Enter an English prompt" }
        require(steps in 8..100)
        require(guidance in 0f..12f)
        val isDdIM = modelId == "ptah32"
        val res = if (isDdIM) 32 else 64
        val maxTokens = if (isDdIM) 24 else 32
        val dir = store.location(modelId)
        val tokenizer = ClipBpeTokenizer(dir)
        val texts = listOf("", prompt).map { tokenizer.encode(it, maxTokens) }
        val ids = LongArray(2 * maxTokens) { i -> texts[i / maxTokens].ids[i % maxTokens] }
        val mask = LongArray(2 * maxTokens) { i -> texts[i / maxTokens].mask[i % maxTokens] }
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(4)
        env.createSession(java.io.File(dir, "clip.onnx").absolutePath, options).use { clipSession ->
            val embeddings: FloatArray
            val idTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(2, maxTokens.toLong()))
            val maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), longArrayOf(2, maxTokens.toLong()))
            try {
                clipSession.run(mapOf("input_ids" to idTensor, "attention_mask" to maskTensor)).use { clipResult ->
                    @Suppress("UNCHECKED_CAST")
                    val hidden = clipResult[0].value as Array<Array<FloatArray>>
                    require(hidden.size == 2 && hidden[0].size == maxTokens && hidden[0][0].size == 512) {
                        "CLIP model returned an incompatible embedding shape"
                    }
                    embeddings = FloatArray(2 * maxTokens * 512) { index ->
                        hidden[index / (maxTokens * 512)][(index / 512) % maxTokens][index % 512]
                    }
                }
            } finally {
                idTensor.close()
                maskTensor.close()
            }

            val random = Random(seed)
            val n = 3 * res * res
            val latent = FloatArray(n) { random.nextGaussian().toFloat() }
            val dup = FloatArray(n * 2)
            val alphas = if (isDdIM) ddimAlphaCumprod() else null
            val stepRatio = if (isDdIM) (1000 / steps) else 0

            env.createSession(java.io.File(dir, "generator.onnx").absolutePath, options).use { genSession ->
                val ctxTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(embeddings),
                    longArrayOf(2, maxTokens.toLong(), 512))
                val ctxMaskTensor = if (!isDdIM) {
                    OnnxTensor.createTensor(env, LongBuffer.wrap(mask), longArrayOf(2, maxTokens.toLong()))
                } else null
                try {
                    for (step in 0 until steps) {
                        coroutineContext.ensureActive()
                        System.arraycopy(latent, 0, dup, 0, n)
                        System.arraycopy(latent, 0, dup, n, n)
                        val xt = OnnxTensor.createTensor(env, FloatBuffer.wrap(dup),
                            longArrayOf(2, 3, res.toLong(), res.toLong()))
                        val timestep = if (isDdIM) {
                            val discrete = ((steps - 1 - step) * stepRatio).toLong()
                            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(discrete, discrete)),
                                longArrayOf(2))
                        } else {
                            val continuous = 1f - step.toFloat() / steps
                            OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(continuous, continuous)),
                                longArrayOf(2))
                        }
                        try {
                            val input = mutableMapOf<String, OnnxTensor>("x" to xt, "t" to timestep, "ctx" to ctxTensor)
                            if (ctxMaskTensor != null) input["ctx_mask"] = ctxMaskTensor
                            genSession.run(input).use { result ->
                                @Suppress("UNCHECKED_CAST")
                                val velocity = result[0].value as Array<Array<Array<FloatArray>>>
                                require(velocity.size == 2 && velocity[0].size == 3 &&
                                    velocity[0][0].size == res) { "Invalid generator output shape" }
                                val pixelArea = res * res
                                val discreteTime = (steps - 1 - step) * stepRatio
                                val alphaT = if (isDdIM) alphas!![discreteTime] else 0.0
                                val prevTime = discreteTime - stepRatio
                                val alphaPrev = if (isDdIM && prevTime >= 0) alphas!![prevTime] else 1.0
                                for (c in 0 until 3) for (y in 0 until res) for (x in 0 until res) {
                                    val index = c * pixelArea + y * res + x
                                    val uncond = velocity[0][c][y][x]
                                    val cond = velocity[1][c][y][x]
                                    val guided = uncond + guidance * (cond - uncond)
                                    if (isDdIM) {
                                        val predicted = ((latent[index] - sqrt(1.0 - alphaT) * guided) /
                                            sqrt(alphaT)).coerceIn(-1.0, 1.0)
                                        latent[index] = (sqrt(alphaPrev) * predicted +
                                            sqrt(1.0 - alphaPrev) * guided).toFloat()
                                    } else {
                                        latent[index] -= guided / steps.toFloat()
                                    }
                                }
                            }
                        } finally {
                            xt.close()
                            timestep.close()
                        }
                        progress(((step + 1) * 100) / steps)
                    }
                } finally {
                    ctxTensor.close()
                    ctxMaskTensor?.close()
                }
            }
            val area = res * res
            val pixels = IntArray(area) { i ->
                fun byte(channel: Int): Int = ((latent[channel * area + i].coerceIn(-1f, 1f) + 1f) *
                    127.5f).roundToInt().coerceIn(0, 255)
                (0xff shl 24) or (byte(0) shl 16) or (byte(1) shl 8) or byte(2)
            }
            Bitmap.createBitmap(pixels, res, res, Bitmap.Config.ARGB_8888)
        }
    }

    private fun ddimAlphaCumprod(): DoubleArray {
        fun alphaBar(t: Double): Double {
            val angle = ((t + 0.008) / 1.008) * PI / 2.0
            return cos(angle) * cos(angle)
        }
        val output = DoubleArray(1000)
        var cumulative = 1.0
        for (t in 0 until 1000) {
            val beta = min(1.0 - alphaBar((t + 1) / 1000.0) / alphaBar(t / 1000.0), 0.999)
            cumulative *= (1.0 - beta)
            output[t] = cumulative
        }
        return output
    }
}
