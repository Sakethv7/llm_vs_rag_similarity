import os
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------------------------------------------------------
# Helper: Authenticate to Hugging Face Hub
# ---------------------------------------------------------------------
def _hf_auth():
    if HF_TOKEN:
        try:
            login(token=HF_TOKEN, add_to_git_credential=False)
            print("✅ Authenticated with Hugging Face Hub")
        except Exception as e:
            print(f"⚠️ Hugging Face login failed: {e}")
    else:
        print("⚠️ HF_TOKEN not found — proceeding without authentication")

# ---------------------------------------------------------------------
# STEP 1: Extract user queries
# ---------------------------------------------------------------------
def extract_user_queries(limit=None):
    print("=" * 60)
    print("STEP 1: Loading User Queries")
    print("=" * 60)
    _hf_auth()

    try:
        print("📥 Loading dataset: lmsys/chatbot_arena_conversations ...")
        ds = load_dataset("lmsys/chatbot_arena_conversations", token=HF_TOKEN)
        split = ds["train"]

        rows = []
        for row in tqdm(split, desc="Extracting user queries"):
            cid = row.get("conversation_id")
            for turn in row["conversation"]:
                if turn.get("role") == "user":
                    rows.append({
                        "user_id": cid,
                        "query_text": turn.get("text", ""),
                        "source": "lmsys_chatbot_arena"
                    })
            if limit and len(rows) >= limit:
                break

        if limit:
            rows = rows[:limit]
        qdf = pd.DataFrame(rows)
        print(f"✅ Loaded {len(qdf)} user queries")
        return qdf

    except Exception as e:
        # If gated dataset fails, fallback to open dataset
        print(f"⚠️ Gated dataset failed ({e}) — falling back to openassistant/oasst1")
        ds = load_dataset("openassistant/oasst1", split="train[:5000]")

        rows = [
            {
                "user_id": r.get("message_id"),
                "query_text": r.get("text", ""),
                "source": "openassistant_oasst1"
            }
            for r in tqdm(ds, desc="Extracting fallback queries")
            if r.get("role") == "prompter"
        ]

        if limit:
            rows = rows[:limit]
        qdf = pd.DataFrame(rows)
        print(f"✅ Loaded {len(qdf)} fallback queries")
        return qdf

# ---------------------------------------------------------------------
# STEP 2: Load or create a prompt library
# ---------------------------------------------------------------------
def build_prompt_library(limit=None):
    print("=" * 60)
    print("STEP 2: Loading Prompt Library")
    print("=" * 60)
    _hf_auth()

    try:
        ds = load_dataset("fka/awesome-chatgpt-prompts")
        pf = ds["train"].to_pandas()
        pf = pf.rename(columns={"act": "prompt_title", "prompt": "canonical_text"})
    except Exception as e:
        print(f"⚠️ Failed to load 'fka/awesome-chatgpt-prompts' ({e}) — using synthetic prompts")
        pf = pd.DataFrame({
            "prompt_title": [f"Prompt {i+1}" for i in range(20)],
            "canonical_text": [f"This is a placeholder prompt #{i+1}" for i in range(20)]
        })

    if limit:
        pf = pf.head(limit)

    pf["prompt_id"] = [f"P{i+1:04d}" for i in range(len(pf))]
    pf["aliases"] = ""
    print(f"✅ Loaded {len(pf)} prompts")
    return pf[["prompt_id", "prompt_title", "canonical_text", "aliases"]]

# ---------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract user queries and prompt library")
    parser.add_argument("--limit", type=int, default=5000, help="Number of queries to load")
    parser.add_argument("--prompt-limit", type=int, default=200, help="Number of prompts to load")
    parser.add_argument("--queries-out", type=str, default="shared/data/queries.csv", help="Output path for queries CSV")
    parser.add_argument("--prompts-out", type=str, default="shared/data/prompts.csv", help="Output path for prompts CSV")

    args = parser.parse_args()

    # Ensure folders exist
    os.makedirs(os.path.dirname(args.queries_out), exist_ok=True)

    qdf = extract_user_queries(limit=args.limit)
    pdf = build_prompt_library(limit=args.prompt_limit)

    qdf.to_csv(args.queries_out, index=False)
    pdf.to_csv(args.prompts_out, index=False)

    print(f"\n✅ Saved queries → {args.queries_out}")
    print(f"✅ Saved prompts → {args.prompts_out}")
