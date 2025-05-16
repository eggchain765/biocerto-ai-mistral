from fastapi import FastAPI

# Crea l'app FastAPI
app = FastAPI()

# Endpoint base GET /
@app.get("/")
def root():
    return {"message": "Biocerto.AI backend attivo"}

# Endpoint POST /ask (placeholder)
@app.post("/ask")
def ask_question():
    return {"answer": "Placeholder attivo"}
