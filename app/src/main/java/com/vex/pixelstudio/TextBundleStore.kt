package com.vex.pixelstudio

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.InputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import kotlin.coroutines.coroutineContext
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/**
 * Installs a complete text-to-sprite ONNX bundle, not a PyTorch .pt checkpoint.
 *
 * Format:
 * - generator.onnx
 * - clip.onnx
 * - vocab.json
 * - merges.txt
 * - manifest.json (model_id, resolution, sampler, max_tokens, sha256)
 *
 * Both graphs are checked before activation; incomplete downloads never count as installed.
 */
class TextBundleStore(private val context: Context) {
    private val base = File(context.filesDir, "text_models").apply { mkdirs() }
    val modelIds = listOf("ptah32", "spritegpt64")
    private val allowed = setOf("generator.onnx", "clip.onnx", "vocab.json", "merges.txt", "manifest.json")

    fun location(id: String): File {
        require(id in modelIds) { "Unsupported model" }
        return File(base, id)
    }

    fun isInstalled(id: String): Boolean {
        if (id !in modelIds) return false
        val dir = location(id)
        return dir.isDirectory && allowed.all { File(dir, it).isFile } &&
            runCatching { JSONObject(File(dir, "manifest.json").readText()).getString("model_id") == id }.getOrDefault(false)
    }

    fun metadata(id: String): JSONObject = JSONObject(File(location(id), "manifest.json").readText())

    fun delete(id: String) {
        if (location(id).exists()) check(location(id).deleteRecursively()) { "Unable to delete model" }
    }

    fun installZip(uri: Uri, expectedId: String) {
        context.contentResolver.openInputStream(uri)?.use { installStream(it, expectedId) }
            ?: error("Unable to open ZIP file")
    }

    /**
     * Downloads directly from this project's release assets and installs after all
     * model files pass SHA-256 + graph validation. Works offline after installation.
     */
    suspend fun downloadAndInstall(id: String, progress: suspend (Long, Long) -> Unit) {
        require(id in modelIds) { "Unsupported model" }
        val release = "https://github.com/Geko7490/VexPixelStudio/releases/download/models-v0.3/"
        val url = release + id + "-bundle.zip"
        withContext(Dispatchers.IO) {
            val partial = File(base, id + ".zip.part")
            try {
                val previous = if (partial.exists()) partial.length() else 0L
                var connection = URL(url).openConnection() as HttpURLConnection
                connection.connectTimeout = 20_000
                connection.readTimeout = 60_000
                connection.instanceFollowRedirects = true
                connection.setRequestProperty("User-Agent", "VexPixelStudio/0.3")
                if (previous > 0) connection.setRequestProperty("Range", "bytes=" + previous + "-")
                var response = connection.responseCode
                var redirects = 0
                while (response in 300..399) {
                    require(redirects++ < 6) { "Too many download redirects" }
                    val next = connection.getHeaderField("Location") ?: error("Missing download redirect URL")
                    require(URL(next).protocol == "https") { "Unsafe model download redirect" }
                    connection.disconnect()
                    connection = URL(next).openConnection() as HttpURLConnection
                    connection.instanceFollowRedirects = true
                    connection.connectTimeout = 20_000
                    connection.readTimeout = 60_000
                    if (previous > 0) connection.setRequestProperty("Range", "bytes=" + previous + "-")
                    response = connection.responseCode
                }
                require(response == 200 || response == 206) { "Model server HTTP " + response }
                val resume = response == 206 && previous > 0
                val contentLength = connection.contentLengthLong
                val total = if (contentLength > 0) contentLength + (if (resume) previous else 0L) else -1L
                var transferred = if (resume) previous else 0L
                try {
                    connection.inputStream.use { input ->
                        FileOutputStream(partial, resume).buffered().use { output ->
                            val buffer = ByteArray(256 * 1024)
                            while (true) {
                                coroutineContext.ensureActive()
                                val read = input.read(buffer)
                                if (read < 0) break
                                transferred += read
                                require(transferred < 1_000_000_000L) { "Model download too large" }
                                output.write(buffer, 0, read)
                                progress(transferred, total)
                            }
                        }
                    }
                } finally { connection.disconnect() }
                require(total <= 0 || transferred == total) { "Incomplete model download" }
                partial.inputStream().use { installStream(it, id) }
                partial.delete()
            } catch (e: Exception) {
                // Preserve the partial download so a supported Range request can resume.
                throw e
            }
        }
    }

    private fun installStream(source: InputStream, expectedId: String) {
        require(expectedId in modelIds)
        val staging = File(base, expectedId + ".installing")
        if (staging.exists()) staging.deleteRecursively()
        check(staging.mkdirs()) { "Cannot create staging directory" }
        try {
            ZipInputStream(source.buffered()).use { zip ->
                    var entry = zip.nextEntry
                    var totalBytes = 0L
                    while (entry != null) {
                        val name = entry.name
                        require(!entry.isDirectory && name in allowed) { "Unexpected file in model bundle: " + name }
                        val target = File(staging, name)
                        require(!target.exists()) { "Duplicate bundle file" }
                        target.outputStream().buffered().use { output ->
                            val buffer = ByteArray(128 * 1024)
                            while (true) {
                                val n = zip.read(buffer)
                                if (n < 0) break
                                totalBytes += n
                                require(totalBytes < 1_500_000_000L) { "Bundle exceeds safety limit" }
                                output.write(buffer, 0, n)
                            }
                        }
                        zip.closeEntry()
                        entry = zip.nextEntry
                    }
                }
            require(allowed.all { File(staging, it).isFile }) { "Missing model bundle files" }
            val manifest = JSONObject(File(staging, "manifest.json").readText())
            require(manifest.getString("model_id") == expectedId) { "Wrong model bundle" }
            val resolution = if (expectedId == "ptah32") 32 else 64
            val sampler = if (expectedId == "ptah32") "ddim" else "euler"
            require(manifest.getInt("resolution") == resolution && manifest.getString("sampler") == sampler) {
                "Incompatible model metadata"
            }
            require(manifest.getInt("max_tokens") == (if (resolution == 32) 24 else 32))
            val hashes = manifest.getJSONObject("sha256")
            for (name in listOf("generator.onnx", "clip.onnx", "vocab.json", "merges.txt")) {
                val expected = hashes.getString(name).lowercase()
                require(expected.matches(Regex("[a-f0-9]{64}")) && expected == sha256(File(staging, name))) {
                    "SHA-256 verification failed for " + name
                }
            }
            val env = OrtEnvironment.getEnvironment()
            env.createSession(File(staging, "generator.onnx").absolutePath, OrtSession.SessionOptions()).use { session ->
                val expected = if (resolution == 32) setOf("x", "t", "ctx") else setOf("x", "t", "ctx", "ctx_mask")
                require(session.inputNames.containsAll(expected) && session.outputNames.contains("velocity")) {
                    "Incompatible generator ONNX graph"
                }
            }
            env.createSession(File(staging, "clip.onnx").absolutePath, OrtSession.SessionOptions()).use { session ->
                require(session.inputNames.containsAll(setOf("input_ids", "attention_mask")) &&
                    session.outputNames.contains("last_hidden_state")) {
                    "Incompatible CLIP ONNX graph"
                }
            }
            val dest = location(expectedId)
            if (dest.exists()) check(dest.deleteRecursively()) { "Cannot replace previous model" }
            check(staging.renameTo(dest)) { "Unable to activate installed model" }
        } finally {
            if (staging.exists()) staging.deleteRecursively()
        }
    }

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val bytes = ByteArray(128 * 1024)
            while (true) {
                val n = input.read(bytes)
                if (n == -1) break
                md.update(bytes, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
