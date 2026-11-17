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
import androidx.compose.foundation.shape.RectangleShape
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
    val pixels = remember(size) { mutableStateListOf<Int>().apply { repeat(size * size) { add(0) } } }
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
                listOf("GENERATE", "EDITOR").forEach { item ->
                    Box(Modifier.weight(1f).background(if (tab == item) violet else ink).clickable { tab = item }.padding(13.dp), contentAlignment = Alignment.Center) {
                        Text(item, color = Color.White, fontSize = 12.sp)
                    }
                }
            }
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(if (tab == "GENERATE") "AI SPRITE GENERATOR" else "PIXEL EDITOR", fontSize = 23.sp, color = ink)
                if (tab == "GENERATE") {
                    Text("REAL ON-DEVICE DIFFUSION · 32×32", fontSize = 11.sp, color = violet)
                    OutlinedTextField(prompt, { prompt = it }, label = { Text("Notes (category model does not read free text)") }, modifier = Modifier.fillMaxWidth(), shape = RectangleShape, maxLines = 2)
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
                    Row(horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        listOf(8, 16, 32).forEach { n ->
                            Button(onClick = { steps = n }, shape = RectangleShape, colors = ButtonDefaults.buttonColors(containerColor = if (steps == n) violet else ink)) { Text("$n steps") }
                        }
                    }
                    OutlinedTextField(seed, { seed = it }, label = { Text("Seed (integer)") }, shape = RectangleShape, modifier = Modifier.fillMaxWidth(), singleLine = true)
                    if (!installed) Button(onClick = { importModel.launch(arrayOf("application/octet-stream", "*/*")) }, shape = RectangleShape, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("IMPORT SPRITE-DIFFUSION ONNX") }
                    Button(onClick = {
                        busy = true
                        percent = 0
                        scope.launch {
                            val result = withContext(Dispatchers.Default) {
                                runCatching {
                                    engine.generate(category, seed.toLongOrNull() ?: 0L, steps) { progress ->
                                        percent = progress
                                    }
                                }
                            }
                            result.onSuccess { image ->
                                size = 32
                                val out = IntArray(1024)
                                image.getPixels(out, 0, 32, 0, 0, 32, 32)
                                out.forEachIndexed { i, value -> pixels[i] = value }
                                image.recycle()
                                status = "Generated 32×32 sprite on device"
                            }.onFailure { status = "Generation failed: ${it.message}" }
                            busy = false
                        }
                    }, shape = RectangleShape, enabled = !busy && installed, modifier = Modifier.fillMaxWidth()) {
                        Text(if (busy) "PROCESSING $percent%" else "GENERATE REAL SPRITE")
                    }
                    if (!installed) Text("Download sprite-150k.onnx from Cyronius/sprite-diffusion on Hugging Face and import it once. Generation works offline afterward.", fontSize = 12.sp, color = ink)
                } else {
                    Text("Canvas ${size}×${size} · draw using your finger", color = ink, fontSize = 12.sp)
                    Row(horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        listOf(16, 32, 64).forEach { n ->
                            Button(onClick = { size = n }, shape = RectangleShape, colors = ButtonDefaults.buttonColors(containerColor = if (size == n) violet else ink)) { Text("${n}×${n}") }
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
                Text(status, fontSize = 12.sp, color = ink)
                Spacer(Modifier.height(12.dp))
            }
        }
    }
}
