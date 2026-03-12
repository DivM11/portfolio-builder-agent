# YFinance Agent

## Overview
The YFinance Agent is a Python-based application designed to help users build personalized finance portfolios using US equities. The application leverages the following technologies:

- **Streamlit**: For building an interactive and user-friendly dashboard.
- **Massive.com (formerly Polygon.io)**: For retrieving stock market data (OHLCV prices, SEC-sourced financial statements).
- **OpenRouter**: For semantic matching and LLM inference to understand user input and provide intelligent analysis.
- **Custom Model Support**: Allows integration of custom models for advanced analysis.

## Features
1. **User Input Handling**: Accepts natural language input to understand user preferences for portfolio creation.
2. **Stock Data Retrieval**: Fetches financial data from Massive.com, including historical prices and financial statements (income statement, balance sheet, cash flow).
3. **Resilient Ticker Fetching**: Shows in-chat progress while fetching each ticker and warns when historical data is unavailable for specific symbols.
4. **Model Selection**: Users can pick different AI models for each pipeline step (Stock Picker, Weight Allocator, Portfolio Analyst, Portfolio Reviewer) from the sidebar. Available models are configured in `config.yml`.
5. **Filtering and Analysis**: Applies user-defined filters and performs backtesting and forecasting.
6. **Interactive Dashboard**: Uses Chat, Historical Prices, and Portfolio tabs, with a post-analysis nudge to open Portfolio results.

## Agent Design

### High-Level Design
- The app follows a two-agent orchestration flow:
   - **PortfolioCreatorAgent** builds the portfolio (ticker recommendation, market data retrieval, summary generation, and weight allocation).
   - **PortfolioEvaluatorAgent** evaluates the produced portfolio, generates analysis, and suggests updates.
- **AgentOrchestrator** controls the workflow loop:
   - `start()` runs Creator then Evaluator.
   - `apply_changes()` re-runs Creator and Evaluator with evaluator feedback.
   - `reject_changes()` finalizes the current portfolio.
- A **human-in-the-loop** step is built into the chat UI:
   - Users can accept or reject suggested changes.
   - Iteration count is bounded by `agents.max_iterations` in `config.yml`.

### Low-Level Design
- **Creator pipeline (per run):**
   1. Validate ticker prompt input/output (runtime-configurable).
   2. Generate recommended tickers from LLM.
   3. Apply follow-up feedback (`add`/`remove`) when present.
   4. Fetch missing ticker data through `TickrDataManager` cache.
   5. Build/reuse summary through `TickrSummaryManager`.
   6. Validate portfolio prompt input/output.
   7. Generate and normalize weights.
   8. Return `AgentResult` with selected tickers, allocation, metadata, and validation details.
- **Evaluator pipeline (per run):**
   1. Validate analysis prompt input/output.
   2. Generate evaluation text and structured suggestions (`add`, `remove`, `reweight`).
   3. Return `AgentResult` with analysis and suggestions.
- **State and caching model:**
   - `TickrDataManager` retains previously fetched ticker payloads across iterations.
   - Cached ticker data is preserved even when a ticker is removed from the final portfolio.
   - `TickrSummaryManager` caches summaries by `(ticker-set, data-version)`.
   - `OrchestratorState` tracks `selected_tickers`, `recommended_tickers`, and `excluded_tickers`.
- **Validation architecture:**
   - Shared abstract output validation strategy with concrete validators for ticker, portfolio, and analysis stages.
   - Runtime toggles under `validations` in `config.yml`:
      - `enabled`, `validate_input`, `validate_output`, `fail_fast`
      - per-stage toggles in `validations.prompts`.
- **Display formatting:**
   - `PortfolioDisplaySummary` formats recommendation changes for chat and portfolio tab display.
   - Suggested changes are rendered in human-readable bullet form instead of raw JSON blobs.

## Project Structure
```
yfinance-agent/
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
   cd yfinance-agent
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
docker run -p 8501:8501 --env-file .secrets yfinance-agent
```

### Massive.com (Polygon.io) Setup
- **API key**: Sign up at [massive.com](https://massive.com) and obtain an API key.
- **Plan requirement**: The **Advanced plan ($199/mo)** is required for financial statement data (income statement, balance sheet). OHLCV price data is available on the free tier.
- The API key is loaded from the environment variable specified in `massive.api.key_env_var` (default: `MASSIVE_API_KEY`).
- Python SDK: `massive` (PyPI) — `pip install -U massive`

### OpenRouter Model Setup
- The app uses configurable models for each pipeline step. Defaults and available choices are set in [config.yml](config.yml) under `openrouter.models` and `openrouter.model_choices`.
- Users can override the model for each task from the sidebar:
  - **Stock Picker** — ticker generation
  - **Weight Allocator** — portfolio weight assignment
  - **Portfolio Analyst** — analysis and evaluation
  - **Portfolio Reviewer** — evaluation and change suggestions
- Default model choices include Claude 3.5 Haiku, Claude 3 Haiku, Gemini Flash 1.5, Gemini 2.0 Flash, Qwen 2.5 72B, and Qwen 2.5 7B.
- OpenRouter settings are grouped under `openrouter.api`, `openrouter.models`, `openrouter.model_choices`, `openrouter.model_labels`, `openrouter.outputs`, `openrouter.temperatures`, and `openrouter.prompts` in [config.yml](config.yml).
- Set the API key via environment variable name specified in `openrouter.api.key_env_var` (default: `OPENROUTER_API_KEY`).

## Using Docker

### Build the Docker Image
```bash
docker build -t yfinance-agent .
```

### Run with Docker Compose (recommended)
```bash
docker compose up                                        # app on :8501
docker compose run --rm test                             # run tests (streams output)
```

### Run with Docker CLI
```bash
# App mode
docker run -p 8501:8501 --env-file .secrets yfinance-agent

# Test mode
docker run --rm yfinance-agent pytest -v --tb=short
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