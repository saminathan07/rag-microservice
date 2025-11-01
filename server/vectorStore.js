// server/vectorStore.js
import fs from 'fs';
import path from 'path';

const VPATH = path.resolve(process.cwd(),'vectors.json');
let vectors = [];

export function loadVectors(){
  vectors = JSON.parse(fs.readFileSync(VPATH,'utf8'));
}

function dot(a,b){ let s=0; for(let i=0;i<a.length;i++) s += a[i]*b[i]; return s; }
function norm(a){ return Math.sqrt(dot(a,a)); }
function cosine(a,b){ return dot(a,b)/(norm(a)*norm(b)); }

export function topK(queryEmb, k=5){
  const scored = vectors.map(v => ({score: cosine(queryEmb, v.embedding), v}));
  scored.sort((a,b)=>b.score-a.score);
  return scored.slice(0,k).map(s => ({score: s.score, id: s.v.id, doc: s.v.doc, chunkIndex: s.v.chunkIndex, text: s.v.text}));
}
