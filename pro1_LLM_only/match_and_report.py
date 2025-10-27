import argparse
import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import json
import time
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

def match_with_llm(query_text, prompts_df, client, model="gpt-4o-mini"):
    """Use OpenAI LLM to find best matching prompt and generate quality scores"""
    
    # Build a concise prompt list for the LLM
    prompt_list = []
    for _, row in prompts_df.iterrows():
        prompt_list.append({
            "id": row["prompt_id"],
            "title": row["prompt_title"],
            "description": row["canonical_text"][:200]  # truncate long descriptions
        })
    
    system_prompt = """You are a prompt matching assistant. Given a user query and a list of prompt templates, 
determine which template (if any) best matches the user's intent. 

Return a JSON object with:
- prompt_id: the ID of the best matching prompt (or null if no good match)
- prompt_title: the title of the matched prompt (or null)
- relevance_score: 0.0-1.0 how well the query matches the prompt's purpose
- completeness_score: 0.0-1.0 how completely the prompt addresses the query
- clarity_score: 0.0-1.0 how clear the user's intent is
- reasoning: brief explanation

Only return a match if relevance_score >= 0.6. Be strict - if unsure, return null."""

    user_prompt = f"""User query: "{query_text}"

Available prompts:
{json.dumps(prompt_list, indent=2)}

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
        
        return {
            "prompt_id": result.get("prompt_id"),
            "prompt_title": result.get("prompt_title"),
            "relevance_score": result.get("relevance_score", 0.0),
            "completeness_score": result.get("completeness_score", 0.0),
            "clarity_score": result.get("clarity_score", 0.0),
            "reasoning": result.get("reasoning", ""),
            "match_method": "llm_only"
        }
    
    except Exception as e:
        print(f"\n⚠️  Error matching query: {e}")
        return {
            "prompt_id": None,
            "prompt_title": None,
            "relevance_score": 0.0,
            "completeness_score": 0.0,
            "clarity_score": 0.0,
            "reasoning": f"Error: {str(e)}",
            "match_method": "error"
        }

def generate_report(matched_df, output_path):
    """Generate comprehensive Excel report with multiple sheets"""
    
    print("\n" + "="*60)
    print("STEP 3: Generating Report")
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
                "clarity_score": "mean"
            }).reset_index()
            faq_coverage.columns = ["Prompt Title", "Usage Count", "Avg Relevance", "Avg Completeness", "Avg Clarity"]
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
    ap = argparse.ArgumentParser(description="Match user queries to prompts using LLM and generate analysis report")
    ap.add_argument("--prompts", required=True, help="Path to prompts CSV")
    ap.add_argument("--queries", required=True, help="Path to queries CSV")
    ap.add_argument("--out", required=True, help="Output Excel file path")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    ap.add_argument("--batch-delay", type=float, default=0.1, help="Delay between API calls (seconds)")
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
    print("PROJECT 1: LLM-ONLY MATCHING")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Approach: Direct LLM matching for all queries")
    
    # Load data
    print("\n" + "="*60)
    print("STEP 1: Loading Data")
    print("="*60)
    prompts = pd.read_csv(args.prompts)
    queries = pd.read_csv(args.queries)
    print(f"✓ Loaded {len(prompts)} prompts")
    print(f"✓ Loaded {len(queries)} queries")

    # Match queries to prompts using LLM
    print("\n" + "="*60)
    print("STEP 2: Matching Queries with LLM")
    print("="*60)
    print("⏱️  This may take a while (each query requires an LLM call)...")
    
    matched_rows = []
    start_time = time.time()
    
    for idx, row in tqdm(queries.iterrows(), total=len(queries), desc="Matching"):
        result = match_with_llm(row["query_text"], prompts, client, model=args.model)
        
        matched_rows.append({
            "user_id": row["user_id"],
            "query_text": row["query_text"],
            "prompt_id": result["prompt_id"],
            "prompt_title": result["prompt_title"],
            "relevance_score": result["relevance_score"],
            "completeness_score": result["completeness_score"],
            "clarity_score": result["clarity_score"],
            "reasoning": result["reasoning"],
            "match_method": result["match_method"]
        })
        
        # Small delay to avoid rate limiting
        time.sleep(args.batch_delay)
    
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