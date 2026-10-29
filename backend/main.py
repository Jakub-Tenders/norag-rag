import os
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import rag

app = FastAPI()


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    docs: list[str]
    cfg: rag.Cfg = rag.Cfg()


def err(status, code, msg):
    return HTTPException(status, {"code": code, "message": msg})


@app.get("/api/documents")
def documents():
    return rag.list_docs()

@app.post("/api/documents")
async def upload(file: UploadFile):
    name = os.path.basename(file.filename or "")
    if not name.endswith((".md", ".txt")):
        raise err(415, "bad_format", "Only .md and .txt files are accepted.")
    try:
        text = (await file.read()).decode("utf-8")
    except UnicodeDecodeError:
        raise err(400, "bad_encoding", "File must be UTF-8 text.")
    if not text.strip():
        raise err(400, "empty_file", "File is empty.")
    rag.save_doc(name, text)
    return {"name": name}

@app.delete("/api/documents/{name}")
def delete(name: str):
    if name not in rag.list_docs():
        raise err(404, "not_found", "Unknown document.")
    rag.delete_doc(name)
    return {"deleted": name}

@app.post("/api/ask")
def ask(body: AskIn):
    if not body.docs:
        raise err(400, "no_documents_selected", "Select at least one document.")
    missing = [d for d in body.docs if d not in rag.list_docs()]
    if missing:
        raise err(404, "not_found", f"Unknown documents: {missing}")
    try:
        return rag.answer(body.question, body.docs, body.cfg)
    except rag.LLMUnavailable:
        raise err(502, "llm_unavailable", "The language model is unreachable. Try again later.")

@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"))