# Portfolio Builder Agent

## Overview

A Streamlit app that uses a **tool-calling LLM agent** to build personalized US equity portfolios. Users describe investment goals in natural language; the agent fetches market data via [Massive.com](https://massive.com), runs analysis through [OpenRouter](https://openrouter.ai)-hosted models, and returns a weighted portfolio with actionable suggestions.

## Features

- **Natural-language portfolio creation** — describe preferences and constraints in plain English.
- **Live market data** — historical prices fetched per ticker with in-chat progress and graceful fallbacks for missing symbols.
- **Switchable LLM models** — choose the active model from the sidebar; options configured in [config.yml](config.yml).
- **Backtesting** — Applies filters and quantitative analysis to the generated portfolio.
- **Tabbed dashboard** — Chat, Historical Prices, and Portfolio views with a post-analysis prompt to review results.

## Agent Design

A single `PortfolioAgent` runs an iterative **tool loop**, calling five tools in sequence:

> `generate_tickers` → `fetch_ticker_data` → `build_summary` → `allocate_weights` → `analyze_portfolio`

The final output is a structured `AgentResult` containing tickers, weights, allocation, analysis text, and suggestions (`add / remove / reweight`).

**Key internals:**
- **Tool-loop execution** — `run()` seeds context, iterates `_run_loop()`, and persists each tool call/result to an event store.
- **Caching** — `TickrDataManager` caches per-ticker payloads; `TickrSummaryManager` caches summaries keyed by ticker set and cache version.
- **Output normalization** — missing fields are backfilled from tool state; suggestions are coerced into a consistent shape.

## Project Structure
```
portfolio-builder-agent/
│
├── docs/               # Documentation files
├── config.yml          # Application configuration
├── docker-compose.yml  # Docker Compose services
├── .secrets.example    # Template for .secrets (gitignored)
├── main.py             # Entry point for the application
├── pyproject.toml      # Poetry configuration file
├── src/
│   ├── config.py       # Configuration loading
│   ├── dashboard.py    # Streamlit dashboard UI
│   ├── data_client.py  # Massive.com (Polygon.io) data fetching
│   ├── llm_validation.py # LLM output validation
│   ├── plots.py        # Plotly chart builders
│   ├── portfolio.py    # Portfolio allocation
│   └── summaries.py    # Data summarization
├── tests/              # Test suite
└── README.md           # Project overview and instructions
```

## Getting Started

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd portfolio-builder-agent
   ```
2. Install dependencies:
   ```bash
   poetry install
   ```

### Running the Application
To start the Streamlit dashboard, run:
```bash
poetry run streamlit run main.py
```

## Configuration and Secrets
- Update [config.yml](config.yml) for model, prompts, and UI text.
- API keys are supplied as **environment variables** at runtime.

### Setting up secrets
Copy the example file and fill in your keys:
```bash
cp .secrets.example .secrets
```
Edit `.secrets`:
```
OPENROUTER_API_KEY=your_openrouter_key_here
MASSIVE_API_KEY=your_massive_com_key_here
```
> `.secrets` is git-ignored.

**Locally (shell):**
```bash
export OPENROUTER_API_KEY="your_openrouter_key_here"
export MASSIVE_API_KEY="your_massive_com_key_here"
poetry run streamlit run main.py
```

**Docker Compose (recommended):**
```bash
docker compose up # reads .secrets automatically
```

**Docker CLI (env-file):**
```bash
docker run -p 8501:8501 --env-file .secrets portfolio-builder-agent
```

### Massive.com (Polygon.io) Setup
- **API key**: Sign up at [massive.com](https://massive.com) and obtain an API key.
- **Plan requirement**: The **Advanced plan ($199/mo)** is required for financial statement data (income statement, balance sheet). OHLCV price data is available on the free tier.
- The API key is loaded from the environment variable specified in `massive.api.key_env_var` (default: `MASSIVE_API_KEY`).
- Python SDK: `massive` (PyPI) — `pip install -U massive`

### OpenRouter Model Setup
- The app uses one active agent model configured under `agent.model` in [config.yml](config.yml).
- Users can switch to any configured option in `openrouter.model_choices` from the sidebar selector.
- OpenRouter settings are grouped under `openrouter.api` and `openrouter.model_choices` in [config.yml](config.yml).
- Set the API key via environment variable name specified in `openrouter.api.key_env_var` (default: `OPENROUTER_API_KEY`).

## Using Docker

### Build the Docker Image
```bash
docker build -t portfolio-builder-agent .
```

### Run with Docker Compose (recommended)
```bash
docker compose up                                        # app on :8501
docker compose run --rm test                             # run tests (streams output)
```

### Run with Docker CLI
```bash
# App mode
docker run -p 8501:8501 --env-file .secrets portfolio-builder-agent

# Test mode
docker run --rm portfolio-builder-agent pytest -v --tb=short
```

### Full rebuild cycle
```bash
docker compose build
docker compose run --rm test
docker compose up
```

## Code Standards
This project follows:
- **PEP8**: Python style guide.
- **SOLID Principles**: For maintainable and scalable code.

## Contributing
Contributions are welcome! Please follow the code standards and submit a pull request.

## License
This project is licensed under the MIT License.