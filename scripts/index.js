/*
 scripts/index.js
 Extended indexer: reads docs/*.txt, docs/*.md, docs/*.pdf, docs/*.docx,
 extracts text, chunks, calls OpenAI embeddings, saves vectors.json.
 Works in ESM projects by using createRequire for CommonJS libs.
*/
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import { createRequire } from 'module';

dotenv.config();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Use require() for CommonJS-only packages
const require = createRequire(import.meta.url);
const pdf = require('pdf-parse');
const mammoth = require('mammoth');

const DOCS_DIR = path.resolve(process.cwd(), 'docs');
const OUT = path.resolve(process.cwd(), 'vectors.json');

const CHUNK_SIZE_WORDS = 400; // ~400 words per chunk
const CHUNK_MIN_WORDS = 30;   // skip tiny chunks
const CHUNK_OVERLAP = 50;     // optional overlap if you want, not used below
const BATCH = 16;

function splitToChunks(text){
  if(!text) return [];
  const words = text.replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
  const chunks = [];
  for(let i=0;i<words.length;i+=CHUNK_SIZE_WORDS){
    const slice = words.slice(i, i+CHUNK_SIZE_WORDS).join(' ');
    if(slice.split(/\s+/).length >= CHUNK_MIN_WORDS) chunks.push(slice);
  }
  return chunks;
}

async function embedMany(texts){
  if(!texts.length) return [];
  // Note: adjust model via .env EMBED_MODEL if you want
  const resp = await client.embeddings.create({
    model: process.env.EMBED_MODEL || 'text-embedding-3-large',
    input: texts
  });
  return resp.data.map(d => d.embedding);
}

async function extractText(fullPath){
  const ext = path.extname(fullPath).toLowerCase();
  try {
    if(ext === '.pdf'){
      const data = fs.readFileSync(fullPath);
      const parsed = await pdf(data);
      // quick preview log (first 200 chars) - helpful while testing
      console.log('  PDF preview:', (parsed.text || '').slice(0,200).replace(/\n+/g,' '));
      return parsed.text || '';
    } else if(ext === '.doc' || ext === '.docx'){
      const buffer = fs.readFileSync(fullPath);
      const res = await mammoth.extractRawText({ buffer });
      console.log('  DOCX preview:', (res.value || '').slice(0,200).replace(/\n+/g,' '));
      return res.value || '';
    } else {
      const txt = fs.readFileSync(fullPath, 'utf8');
      console.log('  TXT preview:', txt.slice(0,200).replace(/\n+/g,' '));
      return txt;
    }
  } catch(err){
    console.error('extractText error for', fullPath, err?.message || err);
    return '';
  }
}

async function main(){
  if(!fs.existsSync(DOCS_DIR)){
    console.error('docs/ folder not found. Put .txt, .md, .pdf, or .docx files into docs/');
    process.exit(1);
  }

  const files = fs.readdirSync(DOCS_DIR).filter(f=>/\.(txt|md|pdf|docx?)$/i.test(f));
  if(files.length === 0){
    console.error('No supported files found in docs/ (txt, md, pdf, docx).');
    process.exit(1);
  }

  const vectors = [];
  for(const f of files){
    console.log('Reading', f);
    const full = path.join(DOCS_DIR, f);
    const text = await extractText(full);
    if(!text || !text.trim()){
      console.warn('No text extracted from', f, '- skipping.');
      continue;
    }

    const chunks = splitToChunks(text);
    if(chunks.length === 0){
      console.warn('No chunks produced for', f, '- skipping.');
      continue;
    }

    for(let i=0;i<chunks.length;i+=BATCH){
      const batch = chunks.slice(i,i+BATCH);
      console.log('Embedding batch', i, 'size', batch.length);
      const embs = await embedMany(batch);
      for(let j=0;j<embs.length;j++){
        vectors.push({
          id: `${f}#${i+j}`,
          doc: f,
          chunkIndex: i+j,
          text: batch[j],
          embedding: embs[j]
        });
      }
      // small sleep to be gentle on rate limits if you need:
      // await new Promise(r => setTimeout(r, 100));
    }
  }

  fs.writeFileSync(OUT, JSON.stringify(vectors, null, 2));
  console.log('Saved', OUT, 'with', vectors.length, 'chunks');
}

main().catch(err=>{ console.error(err); process.exit(1); });
