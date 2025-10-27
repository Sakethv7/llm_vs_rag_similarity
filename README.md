# Prompt Track Analysis

Automated analysis of user queries against a prompt library using LLM-based matching.

## 🚀 Quick Start

### Local Setup

1. **Clone and setup**
```bash
   git clone <your-repo>
   cd PROMPTTRACK_PUBLIC
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
```

2. **Configure secrets**
```bash
   # Copy the example file
   copy .env.example .env  # Windows
   # cp .env.example .env  # Linux/Mac
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=sk-your-actual-key-here
```

3. **Run the pipeline**
```bash
   # Load datasets
   python scripts/load_datasets.py ^
     --queries-out data/queries.csv ^
     --prompts-out data/prompts.csv ^
     --limit 5000 ^
     --prompt-limit 200
   
   # Generate report
   python scripts/match_and_report.py ^
     --prompts data/prompts.csv ^
     --queries data/queries.csv ^
     --out reports/prompt_analysis.xlsx
```

## ⚙️ GitHub Actions Setup

### Setting up Secrets

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add:
   - Name: `OPENAI_API_KEY`
   - Value: `sk-your-openai-key-here`

### Running the Workflow

The workflow runs automatically:
- **Daily at 2 AM UTC** (scheduled)
- **Manually** via the Actions tab

To run manually:
1. Go to **Actions** tab
2. Click **Generate Prompt Analysis Report**
3. Click **Run workflow**
4. (Optional) Adjust query/prompt limits
5. Click **Run workflow**

### Downloading Reports

After the workflow completes:
1. Go to the workflow run
2. Scroll to **Artifacts**
3. Download `prompt-analysis-report-XXX.xlsx`

## 📊 Output

The Excel report contains 4 sheets:
- **UserPromptUsage**: Every query with its matched prompt
- **PromptCoverage**: Which prompts are used and how often
- **UserAdoption**: Per-user adoption rates
- **UncoveredQueries**: Queries that didn't match any prompt

## 🔧 Configuration

Edit `.env` for local runs:
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini  # or gpt-4o for better accuracy
```

Edit `.github/workflows/generate_report.yml` to:
- Change schedule (cron expression)
- Adjust default limits
- Enable/disable auto-commit of reports

## 📁 Project Structure
```
PROMPTTRACK_PUBLIC/
├── .github/
│   └── workflows/
│       └── generate_report.yml
├── scripts/
│   ├── load_datasets.py
│   └── match_and_report.py
├── data/              # Generated CSVs (gitignored)
├── reports/           # Generated Excel files (gitignored)
├── .env.example       # Template for secrets
├── .gitignore
├── requirements.txt
└── README.md
```

## 💡 Tips

- **Cost**: Using gpt-4o-mini costs ~$0.15 per 1M tokens (very cheap for this use case)
- **Speed**: Processing 5000 queries takes ~10-15 minutes
- **Rate Limits**: Adjust `--batch-size` if you hit rate limits
- **Large datasets**: Uncomment `data/*.csv` in `.gitignore` if CSVs are huge

## 🔒 Security

- ✅ Never commit `.env` file
- ✅ Use GitHub Secrets for API keys
- ✅ `.gitignore` prevents accidental commits
- ✅ Use `python-dotenv` to load secrets safely