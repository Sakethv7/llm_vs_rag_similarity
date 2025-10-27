import os, json, csv, hashlib, asyncio
import numpy as np
import pandas as pd
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import AsyncOpenAI

# ---------- Tunables (can override via env in GitHub Actions) ----------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TOPK = int(os.getenv("TOPK", "3"))
EMBED_ONLY_THRESHOLD = float(os.getenv("EMBED_ONLY_THRESHOLD", "0.88"))  # skip LLM if above
CONCURRENCY = int(os.getenv("CONCURRENCY", "10"))  # parallel LLM calls
CACHE_PATH = os.getenv("RAG_CACHE", "shared/.cache_rag_llm.csv")

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------- Helpers ----------
def _hash_key(query, cand_ids):
    h = hashlib.sha256()
    h.update(query.encode("utf-8"))
    h.update((",".join(cand_ids)).encode("utf-8"))
    h.update(MODEL.encode("utf-8"))
    return h.hexdigest()

def load_cache(path):
    if not os.path.exists(path): return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["key"]] = row
    return out

def save_cache(path, cache):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["key","best_id","best_title","score","rationale"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for v in cache.values():
            w.writerow({k: v.get(k,"") for k in fieldnames})

async def embed(texts):
    # returns normalized numpy array
    resp = await client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.stack([np.array(d.embedding, dtype=np.float32) for d in resp.data])
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    return vecs

def cosine_topk(qv, P, k):
    sims = P @ qv  # P and qv normalized -> cosine
    idx = sims.argsort()[-k:][::-1]
    return idx, sims[idx], sims

@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6))
async def llm_compare(query, contexts):
    system = (
        "You will pick which candidate prompt best matches a user's query.\n"
        "Return ONLY JSON with keys: best_id, best_title, score (0..1), rationale.\n"
        "If none are appropriate, set best_id=null, best_title=null, score=0.0."
    )
    user = {"query": query, "candidates": contexts}
    resp = await client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":json.dumps(user, ensure_ascii=False)}
        ],
        response_format={"type":"json_object"}
    )
    return json.loads(resp.choices[0].message.content)

# ---------- Main ----------
async def main_async(prompts_path, queries_path, out_path):
    print("="*60); print("PROJECT 2: RAG (OpenAI embeddings + LLM judge w/ concurrency)"); print("="*60)
    prompts = pd.read_csv(prompts_path)
    queries = pd.read_csv(queries_path)
    print(f"✓ Loaded {len(prompts)} prompts, {len(queries)} queries")

    # Build prompt corpus & embeddings
    corpus = (prompts["prompt_title"].fillna("") + " || " +
              prompts["canonical_text"].fillna("")).tolist()
    ids = prompts["prompt_id"].astype(str).tolist()
    titles = prompts["prompt_title"].fillna("").tolist()
    print("🔹 Embedding prompts (once)…")
    P = await embed(corpus)  # (m,d)

    cache = load_cache(CACHE_PATH)
    results = [None]*len(queries)

    # Prepare tasks with concurrency
    sem = asyncio.Semaphore(CONCURRENCY)

    async def process_one(i, qtext):
        nonlocal results
        if not qtext or not str(qtext).strip():
            results[i] = {"best_id": None, "best_title": None, "score": 0.0, "rationale": "empty"}
            return
        qv = (await embed([qtext]))[0]
        idx, top_sims, _ = cosine_topk(qv, P, TOPK)
        cand_ids = [ids[j] for j in idx]
        cand_titles = [titles[j] for j in idx]
        # if confident, skip LLM
        if float(top_sims[0]) >= EMBED_ONLY_THRESHOLD:
            results[i] = {"best_id": cand_ids[0], "best_title": cand_titles[0],
                          "score": float(top_sims[0]), "rationale": "embed_only"}
            return
        # cache?
        key = _hash_key(qtext, cand_ids)
        if key in cache:
            r = cache[key]
            results[i] = {"best_id": r["best_id"] or None,
                          "best_title": r["best_title"] or None,
                          "score": float(r.get("score", 0.0)),
                          "rationale": r.get("rationale","cache")}
            return
        # LLM judge on top-k (concurrent, guarded by semaphore)
        cands = [{"id": ids[j], "title": titles[j], "text": corpus[j][:320]} for j in idx]
        async with sem:
            data = await llm_compare(qtext, cands)
        # sanitize
        try: score = float(data.get("score", 0.0))
        except: score = 0.0
        score = max(0.0, min(1.0, score))
        out = {"best_id": data.get("best_id"), "best_title": data.get("best_title"),
               "score": score, "rationale": data.get("rationale","llm")}
        cache[key] = {"key": key, "best_id": out["best_id"] or "", "best_title": out["best_title"] or "",
                      "score": str(out["score"]), "rationale": out["rationale"]}
        results[i] = out

    print("🔹 Matching with concurrency …")
    tasks = [process_one(i, str(queries.iloc[i].get("query_text",""))) for i in range(len(queries))]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="RAG-match"):
        await f  # just drive progress bar

    save_cache(CACHE_PATH, cache)

    matched = pd.DataFrame(results)
    df = pd.concat([queries.reset_index(drop=True), matched], axis=1)
    df["IsLibraryQuery"] = df["best_id"].notna() & (df["score"] > 0)

    coverage = (df.groupby("best_id", dropna=False)
                  .agg(questions=("query_text","count"))
                  .reset_index()
                  .merge(prompts.rename(columns={"prompt_id":"best_id"})[["best_id","prompt_title"]],
                         on="best_id", how="left"))
    coverage["used"] = coverage["questions"].fillna(0).astype(int) > 0

    adoption = (df.groupby("user_id")
                  .agg(total=("query_text","count"),
                       used=("IsLibraryQuery", lambda s: int(s.sum())))
                  .reset_index())
    adoption["adoption_pct"] = (adoption["used"]/adoption["total"]).round(3)

    uncovered = df[~df["IsLibraryQuery"]].copy()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xl:
        df.to_excel(xl, index=False, sheet_name="RAG_Matches")
        coverage.sort_values(["used","questions"], ascending=[False, False]).to_excel(xl, index=False, sheet_name="PromptCoverage")
        adoption.sort_values("adoption_pct", ascending=False).to_excel(xl, index=False, sheet_name="UserAdoption")
        uncovered.to_excel(xl, index=False, sheet_name="Uncovered")

    print(f"✅ Report saved → {out_path}")

def main(prompts, queries, out):
    asyncio.run(main_async(prompts, queries, out))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.prompts, args.queries, args.out)
