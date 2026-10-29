import os
import chromadb
from openai import OpenAI, APIError, APIConnectionError
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT, ".env"))
MODEL = "gemini-3.5-flash-lite"
DOCS_DIR = os.path.join(ROOT, "docs")

client = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma = chromadb.PersistentClient(path=os.path.join(ROOT, "chroma_db"))


class Cfg(BaseModel):
    chunk_size: int = Field(800, ge=100, le=4000)  # characters
    overlap_pct: int = Field(10, ge=0, le=50)
    top_k: int = Field(5, ge=1, le=20)


class LLMUnavailable(Exception):
    pass


# Schema the LLM response must match before reaching the UI
class LLMAnswer(BaseModel):
    answerable: bool
    answer: str
    citations: list[int] = []  # 1-based numbers of the passages given in the prompt


# ---------- documents ----------
def list_docs():
    return sorted(f for f in os.listdir(DOCS_DIR) if f.endswith((".md", ".txt")))

def read_doc(name):
    with open(os.path.join(DOCS_DIR, name), encoding="utf-8") as f:
        return f.read()

def save_doc(name, text):
    with open(os.path.join(DOCS_DIR, name), "w", encoding="utf-8") as f:
        f.write(text)
    drop_chunks(name)  # content may have changed: reindexed on next question

def delete_doc(name):
    os.remove(os.path.join(DOCS_DIR, name))
    drop_chunks(name)

def drop_chunks(name):
    for c in chroma.list_collections():
        if c.name.startswith("chunks_"):
            chroma.get_collection(c.name, embedding_function=ef).delete(where={"doc": name})


# ---------- indexing: one collection per chunk size / overlap, as in ask_rag_google.py ----------
def get_col(cfg: Cfg):
    return chroma.get_or_create_collection(f"chunks_{cfg.chunk_size}_{cfg.overlap_pct}",
                                           embedding_function=ef)

def chunk_text(text, cfg: Cfg):
    overlap = cfg.chunk_size * cfg.overlap_pct // 100
    step = cfg.chunk_size - overlap
    return [(i, text[i:i + cfg.chunk_size]) for i in range(0, len(text), step)]

def ensure_indexed(doc, cfg: Cfg):
    col = get_col(cfg)
    if col.get(where={"doc": doc}, limit=1)["ids"]:
        return
    chunks = chunk_text(read_doc(doc), cfg)
    for s in range(0, len(chunks), 5000):
        b = chunks[s:s + 5000]
        col.add(ids=[f"{doc}#{s + j}" for j in range(len(b))],
                documents=[c[1] for c in b],
                metadatas=[{"doc": doc, "start": c[0], "end": c[0] + len(c[1])} for c in b])


# ---------- retrieval ----------
def retrieve(question, docs, cfg: Cfg):
    for d in docs:
        ensure_indexed(d, cfg)
    where = {"doc": docs[0]} if len(docs) == 1 else {"doc": {"$in": docs}}
    r = get_col(cfg).query(query_texts=[question], n_results=cfg.top_k, where=where)
    return [{"chunk_id": i, "doc": m["doc"], "start": m["start"], "end": m["end"], "text": t}
            for i, t, m in zip(r["ids"][0], r["documents"][0], r["metadatas"][0])]


# ---------- generation ----------
PROMPT = """Using only the numbered passages below, answer the question.
If the information is not in the passages, say you don't know (answerable = false).
Reply with ONLY a JSON object: {{"answerable": bool, "answer": string, "citations": [numbers of the passages used]}}.

PASSAGES:
{passages}

QUESTION:
{question}"""

def call_llm(prompt):
    try:
        r = client.chat.completions.create(model=MODEL, timeout=30,
                                           response_format={"type": "json_object"},
                                           messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""
    except (APIError, APIConnectionError) as e:
        raise LLMUnavailable(str(e))

def answer(question, docs, cfg: Cfg):
    hits = retrieve(question, docs, cfg)
    passages = "\n\n".join(f"[{i}] ({h['doc']})\n{h['text']}" for i, h in enumerate(hits, 1))
    raw = call_llm(PROMPT.format(passages=passages, question=question))
    try:
        a = LLMAnswer.model_validate_json(raw)  # invalid JSON or wrong shape -> ValidationError
        if any(c < 1 or c > len(hits) for c in a.citations):
            raise ValueError("cites a passage that was not retrieved")
        if a.answerable and not a.citations:
            raise ValueError("answer without any reference")
    except ValueError as e:  # ValidationError is a ValueError
        return {"status": "invalid_llm_output", "answer": None, "references": [],
                "retrieved": hits, "message": "The model's response was invalid. Retrieved passages are shown instead."}
    if not a.answerable:
        return {"status": "not_in_documents", "answer": None, "references": [],
                "retrieved": hits, "message": "The selected documents don't contain the answer."}
    return {"status": "ok", "answer": a.answer,
            "references": [{"n": c, **hits[c - 1]} for c in sorted(set(a.citations))]}