import os
import logging
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

# ✅ Logging per debug
logging.basicConfig(level=logging.INFO)
logging.info("🚀 Biocerto.AI FastAPI starting...")

# ✅ Istanzia FastAPI
app = FastAPI(title="Biocerto.AI - RAG CPU con Flan-T5")

# ✅ CORS per frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📄 Carica i PDF dalla cartella data/
documents = []
data_path = "data"
os.makedirs(data_path, exist_ok=True)
for file in os.listdir(data_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(data_path, file))
        documents.extend(loader.load())

# ✂️ Dividi i testi
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# 🔎 Embedding con modello potente
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
db = FAISS.from_documents(docs, embedding_model)

# 🧠 Modello LLM leggero per CPU (Flan-T5 base)
os.environ["HF_HOME"] = "/mnt/cache/huggingface"  # se disponibile su Render
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 🔗 RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# 📩 Input schema
class Question(BaseModel):
    query: str

# ✅ Endpoint API
@app.post("/ask")
def ask_question(payload: Question):
    response = qa_chain.run(payload.query)
    return {"answer": response}

# ✅ Endpoint di stato
@app.get("/status")
def status():
    return {"status": "Biocerto.AI RAG attivo", "model": model_id}
