# Biocerto.AI - RAG con Mistral

Sistema di question answering basato su documenti PDF certificati, con:

- Retrieval-Augmented Generation (LangChain + FAISS)
- Modello Mistral 7B Instruct (`mistralai/Mistral-7B-Instruct-v0.2`)
- Embedding potente con `bge-large-en-v1.5`

## 🛠 Requisiti

- Account Hugging Face per scaricare il modello
- Account Render con piano GPU attivo

## 🚀 Deploy su Render

1. Carica questo repo su GitHub
2. Crea un nuovo Web Service su [Render.com](https://render.com)
3. Seleziona:
   - Python
   - GPU Plan
   - Cartella `main.py`
4. Render leggerà `render.yaml` automaticamente

## 📂 Struttura

- `data/`: Inserisci qui i PDF certificati
- `main.py`: Codice dell'API FastAPI
- `/ask`: endpoint POST per domande (input: {"query": "..."})
