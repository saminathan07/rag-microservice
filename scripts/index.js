/*
 scripts/index.js
 Simple indexer: reads docs/*.txt and docs/*.md, chunks, calls OpenAI embeddings,
 saves vectors.json in project root.
*/
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import OpenAI from 'openai';

dotenv.config();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DOCS_DIR = path.resolve(process.cwd(), 'docs');
const OUT = path.resolve(process.cwd(), 'vectors.json');
const CHUNK_SIZE_WORDS = 400; // ~400 words per chunk
const BATCH = 16;

function splitToChunks(text){
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];
  for(let i=0;i<words.length;i+=CHUNK_SIZE_WORDS){
    chunks.push(words.slice(i,i+CHUNK_SIZE_WORDS).join(' '));
  }
  return chunks;
}

async function embedMany(texts){
  if(!texts.length) return [];
  const resp = await client.embeddings.create({
    model: process.env.EMBED_MODEL || 'text-embedding-3-large',
    input: texts
  });
  return resp.data.map(d => d.embedding);
}

async function main(){
  if(!fs.existsSync(DOCS_DIR)){
    console.error('docs/ folder not found. Put .txt or .md files into docs/');
    process.exit(1);
  }
  const files = fs.readdirSync(DOCS_DIR).filter(f=>/\.(txt|md)$/i.test(f));
  const vectors = [];
  for(const f of files){
    console.log('Reading', f);
    const full = path.join(DOCS_DIR, f);
    const text = fs.readFileSync(full, 'utf8');
    const chunks = splitToChunks(text);
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
    }
  }
  fs.writeFileSync(OUT, JSON.stringify(vectors, null, 2));
  console.log('Saved', OUT, 'with', vectors.length, 'chunks');
}

main().catch(err=>{ console.error(err); process.exit(1); });
