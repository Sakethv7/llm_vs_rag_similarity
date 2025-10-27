import argparse
import pandas as pd
from datasets import load_dataset
from pathlib import Path

def extract_user_queries(limit=None):
    """Extract user queries from chatbot arena dataset"""
    print("Loading chatbot arena conversations dataset...")
    ds = load_dataset("lmsys/chatbot_arena_conversations")
    rows = []
    for row in ds["train"]:
        cid = row.get("conversation_id")
        for turn in row["conversation"]:
            if turn.get("role") == "user":
                rows.append({
                    "user_id": cid,
                    "ts": None,
                    "query_text": turn.get("text", ""),
                    "use_case": None,
                    "agent_id": None
                })
        if limit and len(rows) >= limit:
            break
    if limit:
        rows = rows[:limit]
    return pd.DataFrame(rows)

def build_prompt_library(limit=None):
    """Build prompt library from awesome-chatgpt-prompts"""
    print("Loading awesome-chatgpt-prompts dataset...")
    prompts = load_dataset("fka/awesome-chatgpt-prompts")
    df = prompts["train"].to_pandas()
    df = df.rename(columns={"act": "prompt_title", "prompt": "canonical_text"})
    if limit:
        df = df.head(limit)
    df["prompt_id"] = [f"P{i+1:04d}" for i in range(len(df))]
    df["aliases"] = ""
    return df[["prompt_id", "prompt_title", "canonical_text", "aliases"]]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Load and prepare datasets for prompt analysis")
    ap.add_argument("--queries-out", required=True, help="Output path for queries CSV")
    ap.add_argument("--prompts-out", required=True, help="Output path for prompts CSV")
    ap.add_argument("--limit", type=int, default=5000, help="Max number of queries to extract")
    ap.add_argument("--prompt-limit", type=int, default=200, help="Max number of prompts to include")
    args = ap.parse_args()

    # Create output directories if they don't exist
    Path(args.queries_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.prompts_out).parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("STEP 1: Loading User Queries")
    print("="*60)
    qdf = extract_user_queries(limit=args.limit)
    qdf.to_csv(args.queries_out, index=False)
    print(f"✓ Wrote {len(qdf)} queries → {args.queries_out}")

    print("\n" + "="*60)
    print("STEP 2: Loading Prompt Library")
    print("="*60)
    pdf = build_prompt_library(limit=args.prompt_limit)
    pdf.to_csv(args.prompts_out, index=False)
    print(f"✓ Wrote {len(pdf)} prompts → {args.prompts_out}")
    
    print("\n" + "="*60)
    print("✅ Dataset loading complete!")
    print("="*60)
    print(f"\nNext step: Run match_and_report.py with these files")