package com.vex.pixelstudio

import org.json.JSONObject
import java.io.File
import java.nio.charset.StandardCharsets
import java.util.Locale

/**
 * CLIP byte-level BPE tokenizer compatible with openai/clip-vit-base-patch32.
 * Uses the official tokenizer vocab/merges shipped in each validated bundle.
 */
class ClipBpeTokenizer(dir: File) {
    private val vocab: Map<String, Int>
    private val ranks: Map<String, Int>
    private val byteEncoder: Array<String> = Array(256) { "" }
    private val cache = LinkedHashMap<String, List<Int>>()
    private val pattern = Regex("<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+", RegexOption.IGNORE_CASE)
    private val sot: Int
    private val eot: Int

    init {
        val o = JSONObject(File(dir, "vocab.json").readText())
        vocab = o.keys().asSequence().associateWith { o.getInt(it) }
        val merges = File(dir, "merges.txt").readLines()
            .filter { it.isNotBlank() && !it.startsWith("#") }
        ranks = merges.withIndex().associate { it.value.trim() to it.index }
        val original = mutableListOf<Int>()
        original.addAll(33..126)
        original.addAll(161..172)
        original.addAll(174..255)
        val codePoints = original.toMutableList()
        var extra = 0
        for (value in 0..255) {
            if (value !in original) {
                original.add(value)
                codePoints.add(256 + extra++)
            }
        }
        original.forEachIndexed { index, b ->
            byteEncoder[b] = String(Character.toChars(codePoints[index]))
        }
        sot = requireNotNull(vocab["<|startoftext|>"]) { "Missing CLIP start token" }
        eot = requireNotNull(vocab["<|endoftext|>"]) { "Missing CLIP end token" }
    }

    data class Encoded(val ids: LongArray, val mask: LongArray)

    fun encode(text: String, maxTokens: Int): Encoded {
        require(maxTokens >= 3)
        val words = mutableListOf(sot)
        val cleaned = text.lowercase(Locale.ROOT).trim().replace(Regex("\\s+"), " ")
        for (match in pattern.findAll(cleaned)) {
            if (words.size >= maxTokens - 1) break
            val utf8 = match.value.toByteArray(StandardCharsets.UTF_8)
            val encoded = buildString {
                for (b in utf8) append(byteEncoder[b.toInt() and 255])
            }
            val tokenIds = bpe(encoded)
            for (token in tokenIds) {
                if (words.size >= maxTokens - 1) break
                words.add(token)
            }
        }
        words.add(eot)
        val ids = LongArray(maxTokens) { eot.toLong() }
        val mask = LongArray(maxTokens)
        for (index in words.indices) {
            ids[index] = words[index].toLong()
            mask[index] = 1
        }
        return Encoded(ids, mask)
    }

    private fun bpe(word: String): List<Int> {
        cache[word]?.let { return it }
        val symbols = word.codePoints().toArray().map {
            String(Character.toChars(it))
        }.toMutableList()
        if (symbols.isEmpty()) return emptyList()
        symbols[symbols.lastIndex] = symbols.last() + "</w>"
        while (symbols.size > 1) {
            var bestPosition = -1
            var bestRank = Int.MAX_VALUE
            for (i in 0 until symbols.lastIndex) {
                val rank = ranks[symbols[i] + " " + symbols[i + 1]] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestPosition = i
                }
            }
            if (bestPosition < 0) break
            symbols[bestPosition] += symbols[bestPosition + 1]
            symbols.removeAt(bestPosition + 1)
        }
        val result = symbols.map { symbol ->
            vocab[symbol] ?: error("Token not found in official CLIP vocabulary: " + symbol)
        }
        if (cache.size > 2048) cache.clear()
        cache[word] = result
        return result
    }
}
