# 🧠 RAG Microservice

A question-answering system that uses **Retrieval-Augmented Generation (RAG)** with OpenAI’s API 


## 🚀 Features

- Supports **TXT, MD, PDF, DOC, DOCX** documents  
- Uses **OpenAI v8 SDK** for embeddings & generation  
- Stores vectors in **local JSON (vectors.json)**  
- REST API endpoint: `POST /ask`  

## 🧩 Tech Stack

- Node.js + Express  
- OpenAI SDK (v8)  
- pdf-parse + mammoth for file extraction  
- dotenv for configuration  


## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/saminathan07/rag-microservice.git
cd rag-microservice

npm install

crreate  .env

And fill your ccredentials

OPENAI_API_KEY=sk-yourkey
EMBED_MODEL=text-embedding-3-large
GEN_MODEL=gpt-4.1-mini
PORT=3000
TOP_K=5
SCORE_THRESHOLD=0.2


npm run index

npm run dev

http://localhost:3000/ask

curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the contents of the document named simple.txt"}'

