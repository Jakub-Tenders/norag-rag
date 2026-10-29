"""Retrieval baseline. From the repo root:  python eval/run_eval.py [chunk_size overlap_pct top_k]
questions.jsonl, one per line:
  {"id","question","answerable":true,"doc","evidence":"verbatim quote from the doc","expected_answer"}
  {"id","question","answerable":false,"expected_answer":null}
Hit = one of the top-k retrieved chunks (right doc) overlaps the character range of the evidence,
so the number stays comparable when chunk_size / overlap change.
Only answerable questions are scored; unanswerable ones are for checking the failure behaviour."""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
import rag

cfg = rag.Cfg(chunk_size=int(sys.argv[1]) if len(sys.argv) > 1 else 800,
              overlap_pct=int(sys.argv[2]) if len(sys.argv) > 2 else 10,
              top_k=int(sys.argv[3]) if len(sys.argv) > 3 else 5)
qs = [json.loads(l) for l in open(os.path.join(HERE, "questions.jsonl"), encoding="utf-8") if l.strip()]
qs = [q for q in qs if q["answerable"]]
docs = sorted({q["doc"] for q in qs})

results = []
for q in qs:
    start = rag.read_doc(q["doc"]).find(q["evidence"])
    if start < 0:
        print(f"evidence not found for {q['id']}"); continue
    end = start + len(q["evidence"])
    hits = rag.retrieve(q["question"], docs, cfg)
    hit = any(h["doc"] == q["doc"] and h["start"] < end and h["end"] > start for h in hits)
    results.append({"id": q["id"], "hit": hit})

rate = sum(r["hit"] for r in results) / len(results)
out = {"cfg": cfg.model_dump(), "n_questions": len(results), "hit_rate": rate, "results": results}
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
path = os.path.join(HERE, "results", f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg.chunk_size}_{cfg.overlap_pct}_{cfg.top_k}.json")
json.dump(out, open(path, "w"), indent=2)
print(f"{cfg.model_dump()}  hit rate: {rate:.2f} ({sum(r['hit'] for r in results)}/{len(results)})  -> {path}")