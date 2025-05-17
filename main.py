import os
import logging
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# ✅ Logging
logging.basicConfig(level=logging.INFO)
logging.info("🚀 Biocerto.AI FastAPI starting...")

# ✅ FastAPI
app = FastAPI(title="Biocerto.AI - RAG CPU Lazy Load")

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Variabili globali lazy
qa_chain = None
model_id = "google/flan-t5-base"

# ✅ Endpoint base
@app.get("/")
def root():
    return {"status": "Biocerto.AI API attiva"}

# ✅ Endpoint di stato
@app.get("/status")
def status():
    return {
        "status": "RAG pronto" if qa_chain else "In fase di caricamento",
        "model": model_id
    }

# ✅ Schema input
class Question(BaseModel):
    query: str

# ✅ Endpoint domanda
@app.post("/ask")
def ask_question(payload: Question):
    if not qa_chain:
        return {"answer": "⏳ Il sistema è in fase di avvio. Riprova tra qualche secondo."}
    response = qa_chain.run(payload.query)
    return {"answer": response}

# ✅ Startup lazy loading
@app.on_event("startup")
def load_all():
    global qa_chain
    logging.info("⏳ Caricamento RAG in corso...")
    start = time.time()

    # Crea cache se non esiste
    os.environ["HF_HOME"] = "/mnt/cache/huggingface"
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # 📄 PDF
    documents = []
    data_path = "data"
    os.makedirs(data_path, exist_ok=True)
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            documents.extend(loader.load())

    # ✂️ Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # 🔎 FAISS + Embedding
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    db = FAISS.from_documents(docs, embedding_model)

    # 🧠 LLM
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # 🔗 RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    logging.info(f"✅ Sistema pronto in {time.time() - start:.2f} secondi")
