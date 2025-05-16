import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import torch

# ✅ Logging per debug
logging.basicConfig(level=logging.INFO)
logging.info("🚀 Biocerto.AI FastAPI starting...")

# ✅ Istanzia FastAPI
app = FastAPI(title="Biocerto.AI - RAG con Mistral")

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

# 🧠 Definizione placeholder per il modello
llm = None
qa_chain = None

# 🔁 Cache dei modelli Hugging Face
os.environ["HF_HOME"] = "/mnt/cache/huggingface"

@app.on_event("startup")
def load_model():
    global llm, qa_chain
    logging.info("🔁 Caricamento del modello Mistral in corso...")
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    logging.info("✅ Modello Mistral caricato e pronto all'uso.")

# 📩 Input schema
class Question(BaseModel):
    query: str

# ✅ Endpoint API
@app.post("/ask")
def ask_question(payload: Question):
    global qa_chain
    if qa_chain is None:
        return {"answer": "Modello in fase di caricamento. Riprova tra qualche secondo."}
    response = qa_chain.run(payload.query)
    return {"answer": response}
