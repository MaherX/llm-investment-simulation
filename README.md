# LLM Investment Simulation Framework

## Implementing Domain-Specific LLMs for Strategic Investment Decisions: A Retrospective Case Study Comparing AI and Human Expertise

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete implementation for research investigating whether domain-specific Large Language Models can replicate Warren Buffett's investment expertise. The system compares three LLM configurations (OpenAI GPT-4, Anthropic Claude Opus, and consensus approach) against Berkshire Hathaway's documented investment decisions during 2022-2024.

## Research Summary

The study tested three LLM configurations against Berkshire Hathaway's actual investment decisions with the following key findings:

- **Returns**: LLMs achieved 4.72-18.01% versus Berkshire's 42.12%
- **Risk Management**: Superior drawdown control (5-7% versus 14%)
- **Decision Overlap**: Only 6-11% alignment with Buffett's choices
- **Behavioral Biases**: Systematic momentum bias and premature profit-taking identified
- **Enhanced Features**: Behavioral bias correction system with comprehensive academic tracking

## Prerequisites and Installation

### Required API Keys

Create a `.env` file with the following API keys:

```bash
# Required for core functionality
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key  # Required for market data

# Optional for enhanced research capabilities
FINNHUB_API_KEY=your_finnhub_key
NEWS_API_KEY=your_newsapi_key
```

### Installation Process

```bash
# Clone repository
git clone https://github.com/maherhamid/llm-investment-simulation.git
cd llm-investment-simulation

# Install dependencies
pip install -r requirements.txt

# Create environment configuration
cp .env.example .env
# Edit .env with actual API keys
```

### Execution

The system provides multiple simulation modes through interactive selection:

```bash
# Execute main simulation with mode selection
python llm_investment_simulation.py

# Available simulation modes:
# 1. COMPARISON MODE - Executes all three configurations for comparative analysis
# 2. OPENAI ONLY - Utilizes exclusively GPT-4
# 3. ANTHROPIC ONLY - Utilizes exclusively Claude Opus
# 4. BOTH COMBINED - Consensus approach combining both LLMs
# 5. TEST MODE - Minimal API calls for system validation
```

## Project Architecture

```
llm-investment-simulation/
├── llm_investment_simulation.py    # Primary simulation system (v7.0)
├── stock_cache.py                  # Market data caching infrastructure
├── data_filter.py                  # Historical data filtering utilities
├── academic_data_collector.py      # Academic metrics collection system
├── requirements.txt                # Python dependencies specification
├── .env.example                    # Environment variables template
├── README.md                       # Documentation
└── data/
    ├── simulation_data_complete.pkl        # Cached market data
    ├── portfolio_history_*.csv             # Portfolio evolution by mode
    ├── trades_*.csv                        # Transaction records by mode
    ├── daily_equity_*.csv                  # Daily portfolio values
    ├── performance_metrics_*.json          # Performance analysis
    ├── berkshire_convergence_*.csv         # Decision alignment analysis
    ├── decision_log_*.json                 # LLM decisions with rationale
    └── academic_data/
        ├── *_bias_corrections.csv          # Behavioral bias corrections
        ├── *_decision_quality.csv          # Decision quality metrics
        ├── *_theme_analysis.csv            # Theme participation analysis
        ├── *_summary_statistics.json       # Aggregated statistics
        └── *_academic_report.md            # Comprehensive report
```

## System Components

### Investment Decision Engine

- **Buffett Principles Implementation**: Economic moat analysis, management quality assessment, financial strength scoring
- **LLM Integration**: Direct API integration with GPT-4 and Claude Opus for investment decisions
- **Consensus Mechanism**: Combined decision-making requiring agreement between models
- **Behavioral Bias Correction**: Real-time detection and correction of five key biases:
  - Cash drag: Excessive cash holdings
  - Theme blindness: Missing market trends
  - Analysis paralysis: Insufficient trading activity
  - Fixed position sizing: Lack of conviction-based sizing
  - Insufficient selling: Holding underperforming positions

### Data Management Infrastructure

- **Temporal Integrity**: Strict point-in-time data filtering preventing look-ahead bias
- **Comprehensive Sources**: Price data, fundamentals, news sentiment, SEC filings
- **Caching System**: Efficient data storage reducing API calls by approximately 90%
- **Alpha Vantage Integration**: Primary data source with yfinance fallback mechanism

### Portfolio Management System

- **Realistic Constraints**: 
  - Position limits: 20% maximum per position
  - Sector caps: 40% maximum per sector
  - ADV constraints: 10% of average daily volume
  - Minimum cash requirement: 2%
- **Transaction Costs**: Commission modeling, bid-ask spreads, market impact calculations
- **Risk Controls**: Volatility targeting, drawdown management protocols

### Performance Analysis Framework

- **Factor Attribution**: Carhart four-factor model decomposition
- **Risk Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown
- **Benchmark Comparison**: Direct comparison with Berkshire Hathaway and S&P 500
- **Academic Metrics**: 
  - Statistical significance testing
  - Behavioral pattern analysis
  - Decision quality scoring
  - Theme participation tracking

## Configuration Parameters

Primary configuration parameters in `SimulationConfig`:

```python
# Investment Universe Parameters
UNIVERSE_SIZE = 500
MIN_MARKET_CAP = 10_000_000_000  # $10B minimum

# Portfolio Constraints
MAX_POSITION_SIZE = 0.20  # 20% maximum per position
MAX_SECTOR_EXPOSURE = 0.40  # 40% maximum per sector
MIN_CASH = 0.02  # 2% minimum cash requirement

# LLM Configuration
LLM_MODE = 'BOTH'  # Options: 'OPENAI_ONLY', 'ANTHROPIC_ONLY', 'BOTH', 'COMPARE_ALL'
LLM_TEMPERATURE = 0.7
MAX_RESEARCH_PER_DATE = 10  # Number of stocks for deep research

# Decision Schedule (21 quarterly decisions over 3 years)
DECISION_DATES = ['2022-01-15', '2022-03-01', '2022-04-30', ...]
```

## Output Data Structure

### Performance Metrics
- `portfolio_history_{mode}.csv`: Daily portfolio values and holdings composition
- `trades_{mode}.csv`: Complete transaction records with profit/loss calculations
- `performance_metrics_{mode}.json`: Comprehensive performance analysis
- `daily_equity_{mode}.csv`: Daily equity curve data

### Decision Analysis
- `berkshire_convergence_{mode}.csv`: Decision alignment analysis with Berkshire Hathaway
- `decision_log_{mode}.json`: Complete LLM decisions with detailed rationale

### Academic Analysis
- `academic_data/*_bias_corrections.csv`: Behavioral bias correction events
- `academic_data/*_decision_quality.csv`: Comprehensive decision quality metrics
- `academic_data/*_theme_analysis.csv`: Market theme participation analysis
- `academic_data/*_academic_report.md`: Complete research report
- `comparison_results.json`: Head-to-head LLM comparison analysis
- `academic_research_report.json`: Publication-ready statistical analysis

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

To reproduce published results:
1. Utilize commit tagged `v1.0-paper`
2. Configure `DECISION_DATES` to match paper's 21 decision points
3. Set random seed to 42 for all stochastic operations
4. Execute with `LLM_MODE='COMPARE_ALL'`

## Implementation Considerations

### API Cost Estimates
- Full simulation utilizes approximately 50,000 tokens per decision point
- Estimated total cost: $20-50 for complete execution (all modes)
- TEST_MODE available for development with minimal API calls

### Rate Limitations
- Alpha Vantage: 5 calls per minute (free tier) - Required
- OpenAI: Variable based on subscription tier
- Anthropic: Variable based on subscription tier

### Data Requirements
- Storage: Approximately 2GB for cached market data
- Initial download duration: 30-60 minutes
- Subsequent executions utilize cache for improved performance

## Behavioral Bias Correction Framework

The system implements a three-tier correction methodology:

1. **Enhanced Prompting**: Bias-aware instruction sets for LLMs
2. **Decision Validation**: Real-time detection of bias patterns
3. **Mechanical Corrections**: Algorithmic interventions when LLM validation fails

Identified biases and performance impact (basis points):
- Cash drag: 650 bps
- Theme blindness: 780 bps
- Analysis paralysis: 450 bps
- Fixed position sizing: 220 bps
- Insufficient selling: 120 bps

## Troubleshooting

### Common Issues and Solutions

- **API Key Not Found**: Verify `.env` file exists with valid credentials
- **Rate Limit Errors**: Increase delay parameters or upgrade API tier
- **Memory Errors**: Reduce `UNIVERSE_SIZE` or implement batch processing
- **Data Not Found**: Remove cache and re-download (`rm simulation_data_complete.pkl`)
- **Alpha Vantage Errors**: Ensure valid API key is configured (required)

## License

MIT License - See LICENSE file for complete terms

## Contact Information

Dr. cand. Maher Hamid  
Email: mh@amlai.academy  
Institution: La Grande Ecole de Commerce et de Management en alternance, Paris

## Acknowledgments

The author expresses profound gratitude to Allah for the guidance and strength provided throughout this research endeavor.

Special acknowledgment is extended to the author's parents and siblings for their continuous support, understanding, and encouragement during the course of this study. Their faith in this academic pursuit has been invaluable.

The author declares that this research was conducted without external financial support. All costs associated with data acquisition, computational resources, and API access were personally funded.

## Disclaimer

This code is intended for academic research purposes only. It does not constitute financial advice. Past performance does not guarantee future results. Users should conduct their own due diligence before making investment decisions.
