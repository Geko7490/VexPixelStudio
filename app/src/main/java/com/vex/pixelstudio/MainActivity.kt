package com.vex.pixelstudio

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.random.Random

private val ink = Color(0xFF1D1B24)
private val paper = Color(0xFFF5F4EF)
private val violet = Color(0xFF7252C8)
private val categories = listOf("Creature", "Item", "Weapon", "Food", "Effect", "Terrain", "Structure", "Nature")

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { PixelStudio() }
    }
}

@Composable
private fun PixelStudio() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val engine = remember { SpriteInference(context) }
    val textEngine = remember { TextSpriteEngine(context) }
    val bundleStore = remember { TextBundleStore(context) }
    var mode by remember { mutableStateOf("FAST 32") }
    var bundleToImport by remember { mutableStateOf("ptah32") }
    var installedBundlesVersion by remember { mutableIntStateOf(0) }
    val modelManager = remember { ModelManager(context) }
    var activeModel by remember { mutableStateOf(modelManager.activeId) }
    var downloadingId by remember { mutableStateOf<String?>(null) }
    var downloadedBytes by remember { mutableLongStateOf(0L) }
    var downloadTotal by remember { mutableLongStateOf(-1L) }
    var downloadJob by remember { mutableStateOf<Job?>(null) }
    var tab by remember { mutableStateOf("GENERATE") }
    var prompt by remember { mutableStateOf("") }
    var category by remember { mutableIntStateOf(0) }
    var steps by remember { mutableIntStateOf(16) }
    var seed by remember { mutableStateOf(Random.nextLong().toString()) }
    var size by remember { mutableIntStateOf(32) }
    var color by remember { mutableStateOf(Color(0xFF7252C8)) }
    var eraser by remember { mutableStateOf(false) }
    var installed by remember { mutableStateOf(engine.installed()) }
    var busy by remember { mutableStateOf(false) }
    var percent by remember { mutableIntStateOf(0) }
    var status by remember { mutableStateOf("Ready") }
    val pixels = remember { mutableStateListOf<Int>().apply { repeat(32 * 32) { add(0) } } }
    fun resizeCanvas(newSize: Int) {
        if (newSize == size) return
        val oldSize = size
        val copy = pixels.toIntArray()
        val newPixels = IntArray(newSize * newSize) { i ->
            val x = i % newSize
            val y = i / newSize
            if (x < oldSize && y < oldSize) copy[y * oldSize + x] else 0
        }
        pixels.clear()
        pixels.addAll(newPixels.toList())
        size = newSize
    }
    val palette = listOf(Color.Black, Color.White, violet, Color(0xFFEB9445), Color(0xFF369E87), Color(0xFFF1CE57), Color(0xFFDF5974))
    val importModel = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            busy = true
            scope.launch {
                val error = withContext(Dispatchers.IO) { runCatching { engine.importModel(uri) }.exceptionOrNull() }
                installed = engine.installed()
                busy = false
                status = if (error == null) "ONNX model installed" else "Import failed: ${error.message}"
            }
        }
    }
    val importBundle = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null && !busy) {
            val target = bundleToImport
            busy = true
            scope.launch {
                val result = withContext(Dispatchers.IO) {
                    runCatching { bundleStore.installZip(uri, target) }
                }
                installedBundlesVersion++
                status = if (result.isSuccess) target + " text model installed" else
                    "Bundle import failed: " + (result.exceptionOrNull()?.message ?: "Unknown error")
                busy = false
            }
        }
    }
    val export = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("image/png")) { uri ->
        if (uri != null) {
            val copy = pixels.toIntArray()
            val width = size
            scope.launch {
                val error = withContext(Dispatchers.IO) {
                    runCatching {
                        val bitmap = Bitmap.createBitmap(copy, width, width, Bitmap.Config.ARGB_8888)
                        context.contentResolver.openOutputStream(uri)?.use { stream ->
                            check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)) { "PNG export failed" }
                        } ?: error("Cannot write file")
                        bitmap.recycle()
                    }.exceptionOrNull()
                }
                status = if (error == null) "PNG exported" else "Export failed: ${error.message}"
            }
        }
    }
    fun paint(pos: Offset, width: Float, height: Float) {
        val x = (pos.x / width * size).toInt().coerceIn(0, size - 1)
        val y = (pos.y / height * size).toInt().coerceIn(0, size - 1)
        pixels[y * size + x] = if (eraser) 0 else color.toArgb()
    }
    MaterialTheme(colorScheme = lightColorScheme(primary = violet, background = paper, surface = Color.White)) {
        Column(Modifier.fillMaxSize().background(paper)) {
            Row(Modifier.fillMaxWidth().background(ink).padding(17.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("VEX / PIXEL STUDIO", color = Color.White, fontSize = 18.sp)
                Spacer(Modifier.weight(1f))
                Text("LOCAL", color = Color(0xFFCBB8FF), fontSize = 11.sp)
            }
            Row(Modifier.fillMaxWidth().background(ink)) {
                listOf("GENERATE", "EDITOR", "MODELS").forEach { item ->
                    Box(Modifier.weight(1f).background(if (tab == item) violet else ink).clickable { tab = item }.padding(13.dp), contentAlignment = Alignment.Center) {
                        Text(item, color = Color.White, fontSize = 12.sp)
                    }
                }
            }
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(when(tab) { "GENERATE" -> "AI SPRITE GENERATOR"; "EDITOR" -> "PIXEL EDITOR"; else -> "MODEL MANAGER" }, fontSize = 23.sp, color = ink)
                if (tab == "GENERATE") {
                    Text("OFFLINE ONNX IMAGE GENERATION", fontSize = 11.sp, color = violet)
                    Text("GENERATION MODE", fontSize = 12.sp, color = ink)
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.fillMaxWidth()) {
                        listOf("FAST 32", "PROMPT 32", "PROMPT 64").forEach { label ->
                            Button(onClick = {
                                mode = label
                                steps = if (label == "FAST 32") 16 else 50
                            }, shape = RectangleShape, modifier = Modifier.weight(1f),
                                contentPadding = PaddingValues(horizontal = 3.dp, vertical = 6.dp),
                                colors = ButtonDefaults.buttonColors(containerColor = if (mode == label) violet else ink)) {
                                Text(label, fontSize = 10.sp)
                            }
                        }
                    }
                    OutlinedTextField(prompt, { prompt = it },
                        label = { Text(if (mode == "FAST 32") "Notes (Fast mode does not read prompt)" else "English prompt — e.g. a blue bird sprite") },
                        modifier = Modifier.fillMaxWidth(), shape = RectangleShape, maxLines = 2,
                        enabled = mode != "FAST 32")
                    if (mode == "FAST 32") {
                    Text("CATEGORY", fontSize = 12.sp, color = ink)
                    categories.chunked(4).forEach { group ->
                        Row(horizontalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.fillMaxWidth()) {
                            group.forEach { name ->
                                val ix = categories.indexOf(name)
                                Button(onClick = { category = ix }, contentPadding = PaddingValues(horizontal = 4.dp), modifier = Modifier.weight(1f), shape = RectangleShape, colors = ButtonDefaults.buttonColors(containerColor = if (category == ix) violet else ink)) {
                                    Text(name, fontSize = 10.sp)
                                }
                            }
                        }
                    }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        (if (mode == "FAST 32") listOf(8, 16, 32) else if (mode == "PROMPT 32") listOf(20, 30, 50) else listOf(20, 35, 50)).forEach { n ->
                            Button(onClick = { steps = n }, shape = RectangleShape, colors = ButtonDefaults.buttonColors(containerColor = if (steps == n) violet else ink)) { Text("$n steps") }
                        }
                    }
                    OutlinedTextField(seed, { seed = it }, label = { Text("Seed (integer)") }, shape = RectangleShape, modifier = Modifier.fillMaxWidth(), singleLine = true)
                    if (mode == "FAST 32" && !installed) {
                        Button(onClick = { tab = "MODELS" }, shape = RectangleShape,
                            modifier = Modifier.fillMaxWidth()) { Text("SET UP FAST MODEL") }
                        OutlinedButton(onClick = { importModel.launch(arrayOf("application/octet-stream", "*/*")) },
                            shape = RectangleShape, enabled = !busy, modifier = Modifier.fillMaxWidth()) {
                            Text("IMPORT LOCAL 32×32 ONNX")
                        }
                    }
                    if (mode != "FAST 32") {
                        val bundleId = if (mode == "PROMPT 32") "ptah32" else "spritegpt64"
                        val ready = bundleStore.isInstalled(bundleId)
                        Text(if (ready) "TEXT MODEL READY · FULLY OFFLINE" else "TEXT MODEL NOT INSTALLED — IMPORT A VERIFIED ONNX BUNDLE IN MODELS",
                            color = if (ready) violet else Color(0xFF9D5529), fontSize = 11.sp)
                        if (!ready) Button(onClick = { tab = "MODELS" }, shape = RectangleShape,
                            modifier = Modifier.fillMaxWidth()) { Text("OPEN MODEL MANAGER") }
                    }
                    Button(onClick = {
                        busy = true
                        percent = 0
                        scope.launch {
                            val result = runCatching {
                                if (mode == "FAST 32") {
                                    withContext(Dispatchers.Default) {
                                        engine.generate(category, seed.toLongOrNull() ?: 0L, steps) { p ->
                                            percent = p
                                        }
                                    }
                                } else {
                                    val bundleId = if (mode == "PROMPT 32") "ptah32" else "spritegpt64"
                                    textEngine.generate(bundleId, prompt, seed.toLongOrNull() ?: 0L,
                                        steps, if (mode == "PROMPT 32") 5f else 5f) { p ->
                                        withContext(Dispatchers.Main) { percent = p }
                                    }
                                }
                            }
                            result.onSuccess { image ->
                                resizeCanvas(image.width)
                                val values = IntArray(image.width * image.height)
                                image.getPixels(values, 0, image.width, 0, 0, image.width, image.height)
                                values.forEachIndexed { i, c -> pixels[i] = c }
                                image.recycle()
                                status = "Generated " + size + "×" + size + " sprite on your device"
                            }.onFailure {
                                status = "Generation failed: " + (it.message ?: "Unknown error")
                            }
                            busy = false
                        }
                    }, shape = RectangleShape,
                        enabled = !busy && (if (mode == "FAST 32") installed else
                            bundleStore.isInstalled(if (mode == "PROMPT 32") "ptah32" else "spritegpt64") && prompt.isNotBlank()),
                        modifier = Modifier.fillMaxWidth()) {
                        Text(if (busy) "GENERATING " + percent + "%" else "GENERATE REAL SPRITE")
                    }
                    if (mode == "FAST 32" && !installed) Text("Fast mode requires the existing category model. Download it from MODELS.", fontSize = 12.sp, color = ink)
                } else if (tab == "EDITOR") {
                    Text("Canvas ${size}×${size} · draw using your finger", color = ink, fontSize = 12.sp)
                    Row(horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        listOf(16, 32, 64).forEach { n ->
                            Button(onClick = { resizeCanvas(n) }, shape = RectangleShape, colors = ButtonDefaults.buttonColors(containerColor = if (size == n) violet else ink)) { Text("${n}×${n}") }
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        Button(onClick = { eraser = false }, shape = RectangleShape, colors = ButtonDefaults.buttonColors(containerColor = if (!eraser) violet else ink)) { Text("PEN") }
                        Button(onClick = { eraser = true }, shape = RectangleShape, colors = ButtonDefaults.buttonColors(containerColor = if (eraser) violet else ink)) { Text("ERASER") }
                        Button(onClick = { pixels.indices.forEach { pixels[it] = 0 } }, shape = RectangleShape) { Text("CLEAR") }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        palette.forEach { swatch ->
                            Box(Modifier.size(34.dp).background(swatch).border(if (color == swatch) 3.dp else 1.dp, ink).clickable { color = swatch; eraser = false })
                        }
                    }
                }
                if (tab == "MODELS") {
                    Text("Download real models once. Generation runs on device after installation.", fontSize = 13.sp, color = ink)
                    Text("TEXT-TO-SPRITE ONNX BUNDLES", fontSize = 15.sp, color = violet)
                    listOf("ptah32", "spritegpt64").forEach { id ->
                        val built = bundleStore.isInstalled(id)
                        Column(Modifier.fillMaxWidth().border(1.dp, ink).background(Color.White).padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text(if (id == "ptah32") "PTAH X32 · ENGLISH PROMPT" else "SPRITE-GPT 64 · ENGLISH PROMPT",
                                fontSize = 15.sp, color = ink)
                            Text(if (id == "ptah32") "32×32 · DDIM · MIT model license" else
                                "64×64 · Euler · CC-BY-SA 4.0 model license",
                                fontSize = 12.sp, color = Color.DarkGray)
                            Text(if (built) "VERIFIED BUNDLE INSTALLED" else
                                "Requires exported generator + CLIP + tokenizer bundle. Not a raw .pt model.",
                                fontSize = 11.sp, color = if (built) violet else Color(0xFF9D5529))
                            if (downloadingId == id) {
                                val fraction = if (downloadTotal > 0L)
                                    (downloadedBytes.toFloat() / downloadTotal.toFloat()).coerceIn(0f, 1f) else 0f
                                LinearProgressIndicator(progress = { fraction }, modifier = Modifier.fillMaxWidth())
                                Text(if (downloadTotal > 0L)
                                    (downloadedBytes / 1048576).toString() + " / " +
                                        (downloadTotal / 1048576).toString() + " MB"
                                    else (downloadedBytes / 1048576).toString() + " MB",
                                    fontSize = 12.sp, color = ink)
                                OutlinedButton(onClick = { downloadJob?.cancel() }, shape = RectangleShape) {
                                    Text("CANCEL DOWNLOAD")
                                }
                            } else if (!built) {
                                Button(onClick = {
                                    downloadingId = id
                                    downloadedBytes = 0L
                                    downloadTotal = -1L
                                    downloadJob = scope.launch {
                                        try {
                                            bundleStore.downloadAndInstall(id) { bytes, total ->
                                                withContext(Dispatchers.Main) {
                                                    downloadedBytes = bytes
                                                    downloadTotal = total
                                                }
                                            }
                                            installedBundlesVersion++
                                            status = id + " installed and verified"
                                        } catch (e: CancellationException) {
                                            status = "Download paused"
                                        } catch (e: Exception) {
                                            status = "Text model installation failed: " + (e.message ?: "Unknown error")
                                        } finally {
                                            downloadingId = null
                                        }
                                    }
                                }, enabled = !busy && downloadingId == null, shape = RectangleShape,
                                    modifier = Modifier.fillMaxWidth()) {
                                    Text("DOWNLOAD MODEL")
                                }
                                OutlinedButton(onClick = {
                                    bundleToImport = id
                                    importBundle.launch(arrayOf("application/zip", "application/octet-stream", "*/*"))
                                }, enabled = !busy && downloadingId == null, shape = RectangleShape,
                                    modifier = Modifier.fillMaxWidth()) {
                                    Text("IMPORT VERIFIED ONNX ZIP")
                                }
                            } else {
                                Button(onClick = {
                                    bundleStore.delete(id)
                                    installedBundlesVersion++
                                    status = id + " removed"
                                }, enabled = !busy && downloadingId == null, shape = RectangleShape) {
                                    Text("DELETE BUNDLE")
                                }
                            }
                        }
                    }
                    Text("Developer ONNX conversion runs: GitHub Actions → Build real text-to-pixel ONNX model bundles.",
                        fontSize = 11.sp, color = Color.DarkGray)
                    Text("FAST CATEGORY MODELS", fontSize = 15.sp, color = violet)
                    ModelCatalog.models.filter { it.resolution == 32 }.forEach { model ->
                        Column(
                            Modifier.fillMaxWidth().border(1.dp, ink).background(Color.White).padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Text(model.title.uppercase(), fontSize = 16.sp, color = ink)
                            Text(model.note, fontSize = 12.sp, color = Color.DarkGray)
                            if (model.url == null) {
                                Text("NOT AVAILABLE FOR ANDROID — ONNX PORT REQUIRED", color = Color(0xFF9D5529), fontSize = 11.sp)
                            } else {
                                val isInstalled = modelManager.installed(model.id)
                                val selected = activeModel == model.id && isInstalled
                                Text(when {
                                    downloadingId == model.id -> "DOWNLOADING"
                                    selected -> "INSTALLED · ACTIVE"
                                    isInstalled -> "INSTALLED"
                                    else -> "NOT INSTALLED"
                                }, color = violet, fontSize = 11.sp)
                                if (downloadingId == model.id) {
                                    val p = if (downloadTotal > 0) (downloadedBytes.toFloat() / downloadTotal).coerceIn(0f, 1f) else 0f
                                    LinearProgressIndicator(progress = { p }, modifier = Modifier.fillMaxWidth())
                                    Text(if (downloadTotal > 0) "${downloadedBytes / 1048576} / ${downloadTotal / 1048576} MB" else "${downloadedBytes / 1048576} MB", fontSize = 12.sp)
                                    Button(onClick = { downloadJob?.cancel() }, shape = RectangleShape) { Text("CANCEL") }
                                } else if (!isInstalled) {
                                    Button(onClick = {
                                        downloadingId = model.id
                                        downloadedBytes = 0L
                                        downloadTotal = -1L
                                        downloadJob = scope.launch {
                                            try {
                                                modelManager.download(model.id) { bytes, total ->
                                                    withContext(Dispatchers.Main) {
                                                        downloadedBytes = bytes
                                                        downloadTotal = total
                                                    }
                                                }
                                                activeModel = modelManager.activeId
                                                installed = engine.installed()
                                                status = "${model.title} installed and ready"
                                            } catch (e: CancellationException) {
                                                status = "Download paused"
                                            } catch (e: Exception) {
                                                status = "Model download failed: ${e.message}"
                                            } finally { downloadingId = null }
                                        }
                                    }, enabled = downloadingId == null && !busy, shape = RectangleShape) { Text("DOWNLOAD") }
                                } else {
                                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                        Button(onClick = {
                                            modelManager.select(model.id)
                                            activeModel = model.id
                                            installed = engine.installed()
                                            status = "${model.title} selected"
                                        }, enabled = !selected && downloadingId == null && !busy, shape = RectangleShape) { Text(if (selected) "ACTIVE" else "USE MODEL") }
                                        OutlinedButton(onClick = {
                                            modelManager.delete(model.id)
                                            activeModel = modelManager.activeId
                                            installed = engine.installed()
                                            status = "${model.title} removed"
                                        }, enabled = downloadingId == null && !busy, shape = RectangleShape) { Text("DELETE") }
                                    }
                                }
                            }
                        }
                    }
                }
                if (tab != "MODELS") {
                BoxWithConstraints(Modifier.fillMaxWidth().aspectRatio(1f).background(Color.White).border(1.dp, ink)) {
                    val canvasWidth = constraints.maxWidth.toFloat()
                    val canvasHeight = constraints.maxHeight.toFloat()
                    Canvas(Modifier.fillMaxSize().pointerInput(size, eraser, color) {
                        detectTapGestures { paint(it, this.size.width.toFloat(), this.size.height.toFloat()) }
                    }.pointerInput(size, eraser, color) {
                        detectDragGestures { change, _ ->
                            paint(change.position, this.size.width.toFloat(), this.size.height.toFloat())
                            change.consume()
                        }
                    }) {
                        val w = this.size.width / size
                        val h = this.size.height / size
                        for (y in 0 until size) for (x in 0 until size) {
                            val argb = pixels[y * size + x]
                            val background = if ((x / 2 + y / 2) % 2 == 0) Color(0xFFE6E6E6) else Color.White
                            drawRect(background, Offset(x * w, y * h), androidx.compose.ui.geometry.Size(w + 0.3f, h + 0.3f))
                            if ((argb ushr 24) != 0) drawRect(Color(argb), Offset(x * w, y * h), androidx.compose.ui.geometry.Size(w + 0.3f, h + 0.3f))
                        }
                    }
                }
                Button(onClick = { export.launch("vex-pixel-${System.currentTimeMillis()}.png") }, shape = RectangleShape, modifier = Modifier.fillMaxWidth()) { Text("EXPORT PNG") }
                }
                Text(status, fontSize = 12.sp, color = ink)
                Spacer(Modifier.height(12.dp))
            }
        }
    }
}
