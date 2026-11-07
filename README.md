# EDA Workflow (Finished Version)

> This is the finished version of the Week 4 EDA workflow, here as a lifeline if you get stuck. It shows one good way to reach the goal, not the only one, so try it yourself first. The student-facing starter is [eda-workflow-start](https://github.com/Future-Proof-DS/eda-workflow-start).

An AI-powered exploratory data analysis workflow that performs consistent, first-pass analysis of datasets using LangChain and LangGraph. The workflow runs a fixed set of analysis tools, uses an LLM to extract observations after each step, and synthesizes findings into a summary with actionable recommendations.

## How It Works

The workflow follows a sequential process:
1. **Analyze**: Runs a fixed set of predefined analysis tools on the dataset
2. **Observe**: After each tool, the LLM extracts concise observations from the results
3. **Synthesize**: Once all tools have run, the LLM summarizes findings and provides actionable recommendations

This approach combines deterministic pandas-based analysis with LLM-powered interpretation.

## Analysis Tools

The workflow runs four tools, each followed by an observation step. The first two come with the starter. `compute_aggregates` and `analyze_relationships` were added to answer the next questions a data analyst would ask: where do the numbers sit, and which variables depend on each other?

| Tool | What it produces | Why it is here |
| --- | --- | --- |
| `profile_dataset` | Shape, dtypes, column roles and summaries | Sets up everything after it. Column roles decide what later tools are allowed to do with each column. |
| `analyze_missingness` | Nulls and sentinel values per column, complete rows | Nulls alone miss placeholder strings. In `cafe_sales.csv`, `Item` has no nulls but 620 rows (6.36%) are "unknown" or "error". |
| `compute_aggregates` | Totals by category, rows left without a category, monthly trend, date spikes | Shows which groups matter and whether time looks healthy. The spike check flags 2023-02-06, with 491 rows against a typical 25. |
| `analyze_relationships` | Correlations, derived columns, category strength | Correlation only covers numeric pairs and can't tell a real link from a formula. The derived column check shows `Total Spent` is `Quantity * Price Per Unit`, and that 1,166 rows break it. |

### Design choices

- **Column roles instead of dtypes**: pandas sees `Quantity` and `Price Per Unit` as the same kind of number, but only one of them makes sense to add up. `classify_column_roles` sorts columns into IDs, dates, measures, rates, categoricals and free text, using the column name first and the values second. Only measures get summed.
- **Sentinels count as missing**: `mask_sentinels` turns placeholder strings into real missing values, so they are reported by `analyze_missingness` and never show up as categories in the later tools. The rows they leave out are still reported, so nothing disappears silently.
- **Small tables for the LLM**: each tool returns a compact summary rather than the raw data, so the LLM reasons over numbers that pandas computed.
- **Two kinds of observation**: the observation prompt labels each one `[Data quality]` or `[Analysis]`. Both go through the same prompt, so the label keeps "this data is wrong" apart from "this is what the data says" when the findings are synthesized.

The name-based rules are a heuristic. They suit a first pass, but on a new dataset check `column_roles` in the profile output before trusting the aggregates.

## Setup

### Prerequisites

- **Python 3.10 or 3.11**
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):
   
   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:
   
   **Windows**:
   ```powershell
   copy .env.example .env
   ```
   
   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.10

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Python API

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from eda_workflow.eda_workflow import EDAWorkflow

load_dotenv()

# Initialize the workflow with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1000, timeout=90, max_retries=0)
workflow = EDAWorkflow(model=llm)

# Run analysis on a dataset
workflow.invoke_workflow("data/cafe_sales.csv")

# Retrieve results
summary = workflow.get_summary()              # str
recommendations = workflow.get_recommendations()  # list[str]
observations = workflow.get_observations()    # dict[str, list[str]]
results = workflow.get_results()              # dict
```

### Running the Example

```bash
poetry run python example_usage.py
```

This runs a full analysis on the sample dataset and prints the results for each step.

## Project Structure

```
eda-agent/
├── data/
│   └── cafe_sales.csv             # Sample dataset
├── eda_workflow/
│   ├── __init__.py
│   ├── eda_workflow.py             # Main workflow class and graph
│   └── prompts/                   # LLM prompt templates
│       ├── extract_observations_system.txt
│       ├── extract_observations_human.txt
│       ├── synthesize_findings_system.txt
│       └── synthesize_findings_human.txt
├── .env.example                   # Environment variable template
├── example_usage.py               # Example script
├── pyproject.toml                 # Dependencies configuration
├── poetry.lock                    # Locked dependency versions
└── README.md
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.