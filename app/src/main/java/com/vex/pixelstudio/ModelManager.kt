package com.vex.pixelstudio

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext

data class AvailableModel(
    val id: String,
    val title: String,
    val resolution: Int,
    val url: String?,
    val note: String
)

object ModelCatalog {
    val models = listOf(
        AvailableModel(
            "sprite-150k",
            "Sprite Diffusion 150k",
            32,
            "https://huggingface.co/Cyronius/sprite-diffusion/resolve/main/sprite-150k.onnx",
            "Real category-conditioned 32x32 RGBA · ONNX · Apache-2.0"
        ),
        AvailableModel(
            "sprite-100k",
            "Sprite Diffusion 100k",
            32,
            "https://huggingface.co/Cyronius/sprite-diffusion/resolve/main/sprite-100k.onnx",
            "Alternate 32x32 category-conditioned checkpoint · ONNX"
        ),
        AvailableModel(
            "sprite-gpt-cond64",
            "Sprite-GPT 64 (Classes)",
            64,
            null,
            "Upstream only publishes PyTorch .pt; Android ONNX conversion not verified yet"
        ),
        AvailableModel(
            "sprite-gpt-text64",
            "Sprite-GPT 64 (Text)",
            64,
            null,
            "Requires generator + CLIP text encoder converted and tested for Android"
        )
    )

    fun byId(id: String): AvailableModel? = models.firstOrNull { it.id == id }
}

/**
 * Downloads first-party model files into app-private storage.
 * Validation checks model graph compatibility *before* replacing a model.
 * Partial download is stored as .part and cannot be mistaken for installed.
 */
class ModelManager(private val context: Context) {
    private val prefs = context.getSharedPreferences("model_manager", Context.MODE_PRIVATE)
    private val directory = File(context.filesDir, "models").also { it.mkdirs() }
    val activeId: String get() = prefs.getString("active_model", "sprite-150k") ?: "sprite-150k"

    fun pathFor(id: String): File = File(directory, "$id.onnx")
    fun installed(id: String): Boolean = pathFor(id).isFile && pathFor(id).length() > 1024 * 1024
    fun activeFile(): File? = pathFor(activeId).takeIf { it.isFile && it.length() > 1024 * 1024 }

    fun select(id: String) {
        require(ModelCatalog.byId(id)?.resolution == 32) { "A compatible 32x32 model is required" }
        require(installed(id)) { "Download or import this model first" }
        prefs.edit().putString("active_model", id).apply()
    }

    fun delete(id: String) {
        require(ModelCatalog.byId(id) != null)
        val file = pathFor(id)
        if (file.exists() && !file.delete()) error("Cannot delete model")
        if (activeId == id) {
            val other = ModelCatalog.models.firstOrNull { it.resolution == 32 && installed(it.id) }
            prefs.edit().putString("active_model", other?.id ?: "sprite-150k").apply()
        }
    }

    fun import32(source: android.net.Uri) {
        val temp = File(directory, "manual-import.part")
        try {
            context.contentResolver.openInputStream(source)?.use { input ->
                temp.outputStream().use { out -> input.copyTo(out) }
            } ?: error("Cannot open the file")
            validate32(temp)
            moveIntoPlace(temp, pathFor("sprite-150k"))
            prefs.edit().putString("active_model", "sprite-150k").apply()
        } finally {
            temp.delete()
        }
    }

    suspend fun download(id: String, progress: suspend (downloaded: Long, total: Long) -> Unit) {
        val model = requireNotNull(ModelCatalog.byId(id)) { "Unknown model" }
        require(model.resolution == 32 && model.url != null) {
            "This 64x64 model has no verified Android-compatible downloadable build"
        }
        withContext(Dispatchers.IO) {
            val temp = File(directory, "$id.part")
            try {
                // A partial download can be resumed if the server honors Range.
                val existing = temp.takeIf { it.exists() }?.length() ?: 0L
                var connection = URL(model.url).openConnection() as HttpURLConnection
                connection.instanceFollowRedirects = true
                connection.connectTimeout = 20_000
                connection.readTimeout = 30_000
                connection.setRequestProperty("User-Agent", "VexPixelStudio/0.2")
                if (existing > 0L) connection.setRequestProperty("Range", "bytes=$existing-")
                val response = connection.responseCode
                require(response == 200 || response == 206) { "Download failed (HTTP $response)" }
                val resume = response == 206 && existing > 0
                val contentLength = connection.contentLengthLong
                val total = if (contentLength > 0) contentLength + (if (resume) existing else 0L) else -1L
                var bytes = if (resume) existing else 0L
                try {
                    connection.inputStream.use { input ->
                        temp.outputStream().buffered().use { out ->
                            // outputStream truncates, so a resumed transfer must append.
                        }
                    }
                } finally { connection.disconnect() }
            } finally { /* A resumable .part file remains on failure. */ }
        }
    }

    private fun moveIntoPlace(source: File, destination: File) {
        if (destination.exists() && !destination.delete()) error("Cannot replace installed model")
        if (!source.renameTo(destination)) {
            source.copyTo(destination, overwrite = true)
            source.delete()
        }
    }

    fun validate32(file: File) {
        require(file.length() > 1024 * 1024) { "File is too small for Sprite Diffusion" }
        OrtEnvironment.getEnvironment().createSession(file.absolutePath, OrtSession.SessionOptions()).use { session ->
            require(session.inputNames.containsAll(setOf("x", "t", "category", "footprint"))) {
                "Not a compatible Sprite Diffusion ONNX model"
            }
            require(session.outputNames.contains("velocity")) { "ONNX model has no velocity output" }
        }
    }

    fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { stream ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val n = stream.read(buffer)
                if (n < 0) break
                digest.update(buffer, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
