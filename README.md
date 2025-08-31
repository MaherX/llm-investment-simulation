# LLM Investment Simulation Framework

## Implementing Domain-Specific LLMs for Strategic Investment Decisions: A Retrospective Case Study Comparing AI and Human Expertise

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete implementation for the research paper submitted to Digital Finance, Special Issue: Generative AI in Digital Finance (September 2025), investigating whether domain-specific Large Language Models can replicate Warren Buffett's investment expertise.

## Research Summary

We tested three LLM configurations (OpenAI GPT-4, Anthropic Claude Opus, and consensus approach) against Berkshire Hathaway's documented investment decisions during 2022-2024. Key findings:

- Returns: LLMs achieved 4.72-18.01% vs Berkshire's 42.12%
- Risk Management: Superior drawdown control (5-7% vs 14%)
- Decision Overlap: Only 6-11% alignment with Buffett's choices
- Behavioral Biases: Systematic momentum bias and premature profit-taking identified

## Quick Start

### Prerequisites

```bash
# Required API Keys (set in .env file)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key

# Optional APIs for enhanced research
FINNHUB_API_KEY=your_finnhub_key
NEWS_API_KEY=your_newsapi_key
```

### Installation

```bash
# Clone repository
git clone https://github.com/maherhamid/llm-investment-simulation.git
cd llm-investment-simulation

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
cp .env.example .env
# Edit .env with your actual API keys
```

### Running the Simulation

```bash
# Run full comparison (all three modes)
python llm_investment_simulation.py

# Select mode when prompted:
# 1. Comparison Mode (runs all 3 configurations)
# 2. OpenAI Only
# 3. Anthropic Only
# 4. Both Combined
# 5. Test Mode (minimal API calls)
```

## Project Structure

```
llm-investment-simulation/
├── llm_investment_simulation.py    # Main simulation system (v7.0)
├── stock_cache.py                  # Caching system for market data
├── data_filter.py                  # Historical data filtering utilities
├── academic_data_collector.py      # Academic metrics collection
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── README.md                       # This file
└── data/
    ├── simulation_data_complete.pkl     # Cached market data
    ├── portfolio_history_*.csv          # Portfolio evolution
    ├── trades_*.csv                     # Transaction records
    └── performance_metrics_*.json       # Performance analysis
```

## Key Features

### Investment Decision Engine
- Buffett Principles Implementation: Economic moat analysis, management quality assessment, financial strength scoring
- LLM Integration: Direct API calls to GPT-4 and Claude Opus for investment decisions
- Consensus Mechanism: Combined decision-making requiring agreement between models

### Data Management
- Temporal Integrity: Strict point-in-time data to prevent look-ahead bias
- Comprehensive Sources: Price data, fundamentals, news sentiment, SEC filings
- Caching System: Efficient data storage reducing API calls by approximately 90%

### Portfolio Management
- Realistic Constraints: Position limits (20%), sector caps (40%), ADV constraints
- Transaction Costs: Commissions, spreads, market impact modeling
- Risk Controls: Volatility targeting, drawdown management

### Performance Analysis
- Factor Attribution: Carhart four-factor model decomposition
- Risk Metrics: Sharpe, Sortino, Calmar ratios, maximum drawdown
- Benchmark Comparison: Direct comparison with Berkshire Hathaway and S&P 500

## Configuration

Key parameters in SimulationConfig:

```python
# Investment Universe
UNIVERSE_SIZE = 500
MIN_MARKET_CAP = 10_000_000_000  # $10B

# Portfolio Constraints
MAX_POSITION_SIZE = 0.20  # 20% max per position
MAX_SECTOR_EXPOSURE = 0.40  # 40% max per sector
MIN_CASH = 0.02  # 2% minimum cash

# LLM Settings
LLM_MODE = 'BOTH'  # Options: 'OPENAI_ONLY', 'ANTHROPIC_ONLY', 'BOTH'
LLM_TEMPERATURE = 0.7
MAX_RESEARCH_PER_DATE = 10  # Stocks to research deeply
```

## Results Output

The simulation generates comprehensive outputs:

- portfolio_history_{mode}.csv: Daily portfolio values and holdings
- trades_{mode}.csv: All executed trades with P&L
- performance_metrics_{mode}.json: Complete performance analysis
- berkshire_convergence_{mode}.csv: Decision alignment analysis
- decision_log_{mode}.json: All LLM decisions with rationale

## Academic Usage

### Citation

```bibtex
@article{hamid2025llm,
  title={Implementing Domain-Specific LLMs for Strategic Investment Decisions: 
         A Retrospective Case Study Comparing AI and Human Expertise},
  author={Hamid, Maher},
  journal={Digital Finance},
  year={2025},
  note={Special Issue: Generative AI in Digital Finance, Submitted},
  publisher={Springer}
}
```

### Reproducibility

To reproduce paper results:
1. Use commit tagged v1.0-paper
2. Set DECISION_DATES to paper's 21 dates
3. Use seed 42 for random operations
4. Run with LLM_MODE='COMPARE_ALL'

## Important Notes

### API Costs
- Full simulation uses approximately 50,000 tokens per decision point
- Estimated cost: $20-50 for complete run (all modes)
- Use TEST_MODE for development (minimal API calls)

### Rate Limits
- Alpha Vantage: 5 calls/minute (free tier)
- OpenAI: Varies by tier
- Anthropic: Varies by tier

### Data Requirements
- Approximately 2GB storage for cached market data
- Initial data download takes 30-60 minutes
- Subsequent runs use cache (much faster)

## Troubleshooting

### Common Issues

1. No API key found: Check .env file exists and contains valid keys
2. Rate limit errors: Increase delays in config or upgrade API tier
3. Memory errors: Reduce UNIVERSE_SIZE or process in batches
4. Data not found: Delete cache and re-download (rm simulation_data_complete.pkl)

## License

MIT License - see LICENSE file

## Contact

Dr. cand. Maher Hamid  
Email: mh@amlai.academy  
Institution: La Grande Ecole de Commerce et de Management en alternance, Paris

## Acknowledgments

The author expresses profound gratitude to Allah for the guidance and strength provided throughout this research endeavor.

Special thanks are extended to my parents and siblings for their continuous support, understanding, and encouragement during the course of this study. Their faith in my academic pursuits has been invaluable.

The author declares that this research was conducted without external financial support. All costs associated with data acquisition, computational resources, and API access were personally funded by the author.

---

Note: This code is for academic research purposes. Not financial advice. Past performance does not guarantee future results.
