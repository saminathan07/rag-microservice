// server/index.js (complete file with improved /ask handler)
import express from 'express';
import dotenv from 'dotenv';
import bodyParser from 'body-parser';
import { OpenAI } from 'openai';
import { loadVectors, topK } from './vectorStore.js';

dotenv.config();
const app = express();
app.use(bodyParser.json({ limit: '1mb' }));

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

loadVectors();

app.post('/ask', async (req, res) => {
  const start = Date.now();
  try {
    const q = req.body?.question;
    if (!q || typeof q !== 'string' || q.trim().length === 0) {
      return res.status(400).json({ error: 'question required' });
    }
    if (q.length > 2000) return res.status(400).json({ error: 'question too long' });

    // 1) embed question
    const embResp = await client.embeddings.create({
      model: process.env.EMBED_MODEL,
      input: q
    });
    const qEmb = embResp.data[0].embedding;

    // 2) retrieve top candidates
    const rawContexts = topK(qEmb, parseInt(process.env.TOP_K || '5'));

    // --- Boost exact filename mentions so file-specific queries return that doc first ---
    const fileMentionMatch = q.match(/([a-zA-Z0-9_\-]+\.(txt|md|pdf))/i);
    if (fileMentionMatch) {
      const filename = fileMentionMatch[1].toLowerCase();
      for (const rc of rawContexts) {
        if (rc.doc && rc.doc.toLowerCase() === filename) {
          rc.score += 0.6; // strong boost to prioritize the mentioned document
        }
      }
    }

    // 3) filter & re-rank
    const SCORE_THRESHOLD = parseFloat(process.env.SCORE_THRESHOLD || '0.20'); // adjust in .env if needed
    let candidates = rawContexts.filter(c => c.score >= SCORE_THRESHOLD);

    // fallback if everything filtered out
    if (candidates.length === 0) {
      candidates = rawContexts.slice(0, Math.min(3, rawContexts.length));
    }

    // small doc-frequency boost to prefer docs that appear multiple times
    const docCounts = {};
    for (const c of candidates) docCounts[c.doc] = (docCounts[c.doc] || 0) + 1;
    candidates = candidates
      .map(c => ({ ...c, reRankScore: c.score + 0.02 * (docCounts[c.doc] - 1) }))
      .sort((a, b) => b.reRankScore - a.reRankScore);

    // 4) Build strict JSON-only prompt
    const system = `You are a helpful assistant. Use ONLY the provided CONTEXT chunks to answer. If the answer is not contained in the provided context, reply exactly with: {"answer":"I don't know based on the provided documents.","sources":[]} (valid JSON). Do NOT add any extra text outside the JSON.`;
    const contextText = candidates.map((c, idx) => {
      return `[[${idx}]] DOC: ${c.doc}#${c.chunkIndex}\n${c.text}\n---\n`;
    }).join('\n');

    const userPrompt = `Question: ${q}\n\nCONTEXT START\n${contextText}\nCONTEXT END\n\nINSTRUCTIONS:\n- Answer concisely using ONLY the context above.\n- Provide a JSON object with two keys: \"answer\" (string) and \"sources\" (array).\n- \"sources\" must be an array of objects {\"doc\":\"<filename>\",\"chunkIndex\":<n>,\"score\":<float>} referencing only chunks from the CONTEXT.\n- If context does not contain the answer, return: {\"answer\":\"I don't know based on the provided documents.\",\"sources\":[]}.\n- Output must be valid JSON and nothing else.`;

    // 5) Call Responses API (Responses uses max_output_tokens)
    const completion = await client.responses.create({
      model: process.env.GEN_MODEL,
      input: [
        { role: 'system', content: system },
        { role: 'user', content: userPrompt }
      ],
      max_output_tokens: 512,
      temperature: 0.0
    });

    // extract text
    let raw = completion.output_text ?? (() => {
      if (Array.isArray(completion.output)) {
        return completion.output.map(o => {
          if (o.content) return o.content.map(c => c.text || '').join('');
          return '';
        }).join('');
      }
      return JSON.stringify(completion);
    })();

    raw = (typeof raw === 'string') ? raw.trim() : String(raw);

    // try to parse JSON
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (parseErr) {
      // Return debugging info so you can inspect model output
      const latency = Date.now() - start;
      return res.status(500).json({
        error: 'model_response_not_json',
        raw,
        parseError: String(parseErr),
        used_contexts: candidates.map(c => ({ doc: c.doc, chunkIndex: c.chunkIndex, score: c.score, reRankScore: c.reRankScore })),
        latency_ms: latency
      });
    }

    // validate shape
    if (typeof parsed !== 'object' || typeof parsed.answer !== 'string' || !Array.isArray(parsed.sources)) {
      const latency = Date.now() - start;
      return res.status(500).json({
        error: 'invalid_model_json_shape',
        parsed,
        used_contexts: candidates.map(c => ({ doc: c.doc, chunkIndex: c.chunkIndex, score: c.score })),
        latency_ms: latency
      });
    }

    // success: return parsed answer + sources + used_contexts for audit
    const latency = Date.now() - start;
    return res.json({
      answer: parsed.answer,
      sources: parsed.sources,
      used_contexts: candidates.map(c => ({ doc: c.doc, chunkIndex: c.chunkIndex, score: c.score, reRankScore: c.reRankScore })),
      latency_ms: latency
    });

  } catch (err) {
    console.error('ERROR /ask', err);
    return res.status(500).json({ error: 'server_error', details: err?.message || String(err) });
  }
});

app.listen(process.env.PORT || 3000, () => console.log('server started'));
