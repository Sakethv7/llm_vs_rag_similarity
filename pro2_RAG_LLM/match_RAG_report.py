import argparse
import os
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import json
import time
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

def get_embedding(text, client, model="text-embedding-3-small"):
    """Get embedding vector for text"""
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️  Embedding error: {e}")
        return None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_search(query_embedding, prompt_embeddings, top_k=3):
    """Find top-k most similar prompts using cosine similarity"""
    similarities = []
    for idx, prompt_emb in enumerate(prompt_embeddings):
        if prompt_emb is not None and query_embedding is not None:
            sim = cosine_similarity(query_embedding, prompt_emb)
            similarities.append((idx, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def llm_judge(query_text, candidates, client, model="gpt-4o-mini"):
    """Use LLM to judge which candidate is the best match"""
    
    if not candidates:
        return {
            "prompt_id": None,
            "prompt_title": None,
            "relevance_score": 0.0,
            "completeness_score": 0.0,
            "clarity_score": 0.0,
            "similarity_score": 0.0,
            "reasoning": "No candidates found"
        }
    
    candidate_text = "\n".join([
        f"{i+1}. {c['title']} (similarity: {c['similarity']:.3f})\n   {c['description'][:150]}..."
        for i, c in enumerate(candidates)
    ])
    
    system_prompt = """You are a prompt matching judge. Given a user query and semantically similar candidate prompts,
determine which candidate (if any) truly matches the user's intent.

Return a JSON object with:
- candidate_number: 1, 2, 3, or null if no good match
- relevance_score: 0.0-1.0 how well the query matches the prompt's purpose
- completeness_score: 0.0-1.0 how completely the prompt addresses the query
- clarity_score: 0.0-1.0 how clear the user's intent is
- reasoning: brief explanation

Only return a match if relevance_score >= 0.6. Be strict."""

    user_prompt = f"""User query: "{query_text}"

Candidate prompts:
{candidate_text}

Return ONLY a JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        candidate_num = result.get("candidate_number")
        if candidate_num and 1 <= candidate_num <= len(candidates):
            selected = candidates[candidate_num - 1]
            return {
                "prompt_id": selected["id"],
                "prompt_title": selected["title"],
                "relevance_score": result.get("relevance_score", 0.0),
                "completeness_score": result.get("completeness_score", 0.0),
                "clarity_score": result.get("clarity_score", 0.0),
                "similarity_score": selected["similarity"],
                "reasoning": result.get("reasoning", "")
            }
        else:
            return {
                "prompt_id": None,
                "prompt_title": None,
                "relevance_score": 0.0,
                "completeness_score": 0.0,
                "clarity_score": 0.0,
                "similarity_score": 0.0,
                "reasoning": result.get("reasoning", "No good match")
            }
    
    except Exception as e:
        return {
            "prompt_id": None,
            "prompt_title": None,
            "relevance_score": 0.0,
            "completeness_score": 0.0,
            "clarity_score": 0.0,
            "similarity_score": 0.0,
            "reasoning": f"Error: {str(e)}"
        }

def generate_report(matched_df, output_path):
    """Generate comprehensive Excel report with multiple sheets"""
    
    print("\n" + "="*60)
    print("STEP 4: Generating Report")
    print("="*60)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Sheet 1: Summary
        matched_queries = matched_df[matched_df["prompt_id"].notna()]
        summary_data = {
            "Metric": [
                "Total Queries",
                "Matched Queries",
                "Unmatched Queries",
                "Match Rate",
                "Avg Relevance Score",
                "Avg Completeness Score",
                "Avg Clarity Score",
                "Avg Similarity Score",
                "Unique Prompts Used"
            ],
            "Value": [
                len(matched_df),
                len(matched_queries),
                len(matched_df[matched_df["prompt_id"].isna()]),
                f"{(len(matched_queries) / len(matched_df) * 100):.1f}%" if len(matched_df) > 0 else "0.0%",
                f"{matched_queries['relevance_score'].mean():.3f}" if len(matched_queries) > 0 else "0.000",
                f"{matched_queries['completeness_score'].mean():.3f}" if len(matched_queries) > 0 else "0.000",
                f"{matched_queries['clarity_score'].mean():.3f}" if len(matched_queries) > 0 else "0.000",
                f"{matched_queries['similarity_score'].mean():.3f}" if len(matched_queries) > 0 else "0.000",
                matched_df["prompt_id"].nunique()
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
        
        # Sheet 2: Full Analysis
        matched_df.to_excel(writer, sheet_name="FullAnalysis", index=False)
        
        # Sheet 3: FAQ Coverage
        if len(matched_queries) > 0:
            faq_coverage = matched_queries.groupby("prompt_title").agg({
                "query_text": "count",
                "relevance_score": "mean",
                "completeness_score": "mean",
                "clarity_score": "mean",
                "similarity_score": "mean"
            }).reset_index()
            faq_coverage.columns = ["Prompt Title", "Usage Count", "Avg Relevance", "Avg Completeness", "Avg Clarity", "Avg Similarity"]
            faq_coverage = faq_coverage.sort_values("Usage Count", ascending=False)
            faq_coverage.to_excel(writer, sheet_name="FAQCoverage", index=False)
        else:
            pd.DataFrame({"Message": ["No matched queries found"]}).to_excel(writer, sheet_name="FAQCoverage", index=False)
        
        # Sheet 4: User Adoption
        user_adoption = matched_df.groupby("user_id").agg({
            "query_text": "count",
            "prompt_id": lambda x: x.notna().sum(),
            "relevance_score": "mean"
        }).reset_index()
        user_adoption.columns = ["User ID", "Total Queries", "Matched Queries", "Avg Relevance"]
        user_adoption["Match Rate"] = (user_adoption["Matched Queries"] / user_adoption["Total Queries"] * 100).round(1)
        user_adoption.to_excel(writer, sheet_name="UserAdoption", index=False)
        
        # Sheet 5: Quality Distribution
        if len(matched_queries) > 0:
            quality_bins = pd.cut(matched_queries["relevance_score"], 
                                 bins=[0, 0.6, 0.75, 0.85, 1.0], 
                                 labels=["Low (0.6-0.75)", "Medium (0.75-0.85)", "High (0.85-1.0)"])
            quality_dist = quality_bins.value_counts().reset_index()
            quality_dist.columns = ["Quality Range", "Count"]
            quality_dist.to_excel(writer, sheet_name="QualityDistribution", index=False)
        else:
            pd.DataFrame({"Message": ["No matched queries to analyze"]}).to_excel(writer, sheet_name="QualityDistribution", index=False)
        
        # Sheet 6: Unmatched Queries
        unmatched = matched_df[matched_df["prompt_id"].isna()][["user_id", "query_text", "reasoning"]]
        unmatched.to_excel(writer, sheet_name="UnmatchedQueries", index=False)
    
    print(f"✓ Report saved to: {output_path}")

def main():
    ap = argparse.ArgumentParser(description="Match queries using RAG + LLM hybrid approach")
    ap.add_argument("--prompts", required=True, help="Path to prompts CSV")
    ap.add_argument("--queries", required=True, help="Path to queries CSV")
    ap.add_argument("--out", required=True, help="Output Excel file path")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    ap.add_argument("--embedding-model", default="text-embedding-3-small", help="Embedding model")
    ap.add_argument("--top-k", type=int, default=3, help="Number of candidates to retrieve")
    args = ap.parse_args()

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ OPENAI_API_KEY not found!\n"
            "Please set it in one of these ways:\n"
            "  1. Create a .env file with: OPENAI_API_KEY=sk-...\n"
            "  2. Set environment variable: export OPENAI_API_KEY=sk-...\n"
            "  3. (GitHub Actions) Add it as a repository secret"
        )
    
    client = OpenAI(api_key=api_key)

    print("\n" + "="*60)
    print("PROJECT 2: RAG + LLM HYBRID")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Approach: Semantic search → LLM judge (top-{args.top_k})")
    
    # Load data
    print("\n" + "="*60)
    print("STEP 1: Loading Data")
    print("="*60)
    prompts = pd.read_csv(args.prompts)
    queries = pd.read_csv(args.queries)
    print(f"✓ Loaded {len(prompts)} prompts")
    print(f"✓ Loaded {len(queries)} queries")

    # Generate prompt embeddings
    print("\n" + "="*60)
    print("STEP 2: Generating Prompt Embeddings")
    print("="*60)
    prompt_embeddings = []
    for _, row in tqdm(prompts.iterrows(), total=len(prompts), desc="Embedding prompts"):
        text = f"{row['prompt_title']}: {row['canonical_text']}"
        emb = get_embedding(text, client, model=args.embedding_model)
        prompt_embeddings.append(emb)
    print(f"✓ Generated {len(prompt_embeddings)} embeddings")

    # Match queries
    print("\n" + "="*60)
    print("STEP 3: Matching Queries (RAG + LLM)")
    print("="*60)
    print("⚡ Fast semantic search + selective LLM judging...")
    
    matched_rows = []
    start_time = time.time()
    
    for idx, row in tqdm(queries.iterrows(), total=len(queries), desc="Matching"):
        # Get query embedding
        query_emb = get_embedding(row["query_text"], client, model=args.embedding_model)
        
        if query_emb is None:
            matched_rows.append({
                "user_id": row["user_id"],
                "query_text": row["query_text"],
                "prompt_id": None,
                "prompt_title": None,
                "relevance_score": 0.0,
                "completeness_score": 0.0,
                "clarity_score": 0.0,
                "similarity_score": 0.0,
                "reasoning": "Failed to generate embedding",
                "match_method": "rag_llm_error"
            })
            continue
        
        # Semantic search for top candidates
        top_matches = semantic_search(query_emb, prompt_embeddings, top_k=args.top_k)
        
        # Build candidate list
        candidates = []
        for prompt_idx, similarity in top_matches:
            candidates.append({
                "id": prompts.iloc[prompt_idx]["prompt_id"],
                "title": prompts.iloc[prompt_idx]["prompt_title"],
                "description": prompts.iloc[prompt_idx]["canonical_text"],
                "similarity": similarity
            })
        
        # LLM judges the candidates
        result = llm_judge(row["query_text"], candidates, client, model=args.model)
        
        matched_rows.append({
            "user_id": row["user_id"],
            "query_text": row["query_text"],
            "prompt_id": result["prompt_id"],
            "prompt_title": result["prompt_title"],
            "relevance_score": result["relevance_score"],
            "completeness_score": result["completeness_score"],
            "clarity_score": result["clarity_score"],
            "similarity_score": result["similarity_score"],
            "reasoning": result["reasoning"],
            "match_method": "rag_llm"
        })
        
        # Small delay to avoid rate limiting
        time.sleep(0.05)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Completed in {elapsed/60:.1f} minutes ({elapsed/len(queries):.2f}s per query)")
    
    matched_df = pd.DataFrame(matched_rows)
    
    # Generate report
    generate_report(matched_df, args.out)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print(f"📊 Report: {args.out}")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()