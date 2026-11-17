package com.vex.pixelstudio

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RectangleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

private val Ink = Color(0xFF16181D)
private val Paper = Color(0xFFF5F4EF)
private val Accent = Color(0xFF6E50C7)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { PixelStudio() }
    }
}

@Composable
private fun PixelStudio() {
    var prompt by remember { mutableStateOf("") }
    var resolution by remember { mutableIntStateOf(32) }
    var selectedTab by remember { mutableStateOf("Generate") }
    var selectedColor by remember { mutableStateOf(Color(0xFF6E50C7)) }
    val pixels = remember(resolution) { mutableStateListOf<Color>().apply { repeat(resolution * resolution) { add(Color.Transparent) } } }
    MaterialTheme(colorScheme = lightColorScheme(primary = Accent, background = Paper, surface = Color.White)) {
        Column(Modifier.fillMaxSize().background(Paper)) {
            Row(Modifier.fillMaxWidth().background(Ink).padding(18.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("VEX / PIXEL STUDIO", color = Color.White, fontSize = 19.sp)
                Spacer(Modifier.weight(1f))
                Text("OFFLINE", color = Color(0xFFBCA9FF), fontSize = 11.sp)
            }
            Row(Modifier.fillMaxWidth()) {
                listOf("Generate", "Editor", "Projects").forEach { tab ->
                    Box(Modifier.weight(1f).clickable { selectedTab = tab }.background(if (selectedTab == tab) Accent else Ink).padding(13.dp), contentAlignment = Alignment.Center) {
                        Text(tab.uppercase(), color = Color.White, fontSize = 12.sp)
                    }
                }
            }
            LazyColumn(Modifier.fillMaxSize().padding(18.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                item {
                    Text(if (selectedTab == "Generate") "CREATE PIXEL ART" else if (selectedTab == "Editor") "PIXEL EDITOR" else "PROJECT LIBRARY", fontSize = 24.sp, color = Ink)
                    Text("Sharp pixels. Local-first workflow.", color = Color.DarkGray, fontSize = 13.sp)
                }
                if (selectedTab == "Generate") {
                    item { OutlinedTextField(value = prompt, onValueChange = { prompt = it }, label = { Text("Prompt (AI model not installed)") }, modifier = Modifier.fillMaxWidth(), shape = RectangleShape, minLines = 2) }
                    item {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            listOf(16, 32, 64).forEach { size ->
                                Button(onClick = { resolution = size }, shape = RectangleShape, colors = ButtonDefaults.buttonColors(containerColor = if (resolution == size) Accent else Ink)) { Text("${size}×${size}") }
                            }
                        }
                    }
                    item {
                        Button(onClick = {}, enabled = false, shape = RectangleShape, modifier = Modifier.fillMaxWidth()) { Text("GENERATE — MODEL SETUP REQUIRED") }
                        Text("No fake output: AI generation stays disabled until a real on-device model is integrated.", fontSize = 12.sp, color = Color.DarkGray)
                    }
                }
                if (selectedTab != "Projects") {
                    item {
                        Box(Modifier.fillMaxWidth().aspectRatio(1f).background(Color.White).border(1.dp, Ink)) {
                            Canvas(Modifier.fillMaxSize()) {
                                val step = size.width / resolution
                                for (y in 0 until resolution) for (x in 0 until resolution) {
                                    val index = y * resolution + x
                                    val checker = if ((x + y) % 2 == 0) Color(0xFFE9E9E9) else Color.White
                                    drawRect(checker, Offset(x * step, y * step), androidx.compose.ui.geometry.Size(step, step))
                                    if (pixels[index] != Color.Transparent) drawRect(pixels[index], Offset(x * step, y * step), androidx.compose.ui.geometry.Size(step, step))
                                }
                                for (i in 0..resolution) {
                                    val pos = i * step
                                    drawLine(Color(0x1A000000), Offset(pos, 0f), Offset(pos, size.height), 0.5f)
                                    drawLine(Color(0x1A000000), Offset(0f, pos), Offset(size.width, pos), 0.5f)
                                }
                            }
                        }
                    }
                    item {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            listOf(Color.Black, Color.White, Accent, Color(0xFFEB9445), Color(0xFF369E87)).forEach { color ->
                                Box(Modifier.size(42.dp).background(color).border(if (selectedColor == color) 3.dp else 1.dp, Ink).clickable { selectedColor = color })
                            }
                        }
                    }
                } else {
                    item { Text("Projects will appear here after local persistence is implemented. No sample projects are fabricated.", color = Color.DarkGray) }
                }
            }
        }
    }
}
