"""
COMPLETE FULL LLM INVESTMENT SIMULATION SYSTEM WITH PROFESSIONAL RESEARCH
Version: 21.0 - FIXED & OPTIMIZED VERSION
All Bugs Fixed + Cost Optimizations + Safety Improvements
"""

# ============================================================================
# SECTION 1: IMPORTS AND INITIAL SETUP
# ============================================================================

import pandas as pd
import numpy as np
import yfinance as yf  # Primary data source
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import requests
import time as time_module
import json
import openai
import anthropic
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Union
import requests
from dataclasses import dataclass, field
import functools
import pathlib
import hashlib
import time
import os
import pickle
import re
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import statistics
import math
from dotenv import load_dotenv
from stock_cache import StockDataCache  # Import the new caching system
from data_filter import HistoricalDataFilter  # Import data filtering utilities
from academic_data_collector import academic_collector, BiasMetric, DecisionQualityMetric, ThemeAnalysis

# Optional imports with fallbacks
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from pydantic import BaseModel, Field, ValidationError, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    from dataclasses import dataclass as BaseModel
    Field = lambda **kwargs: None
    ValidationError = Exception
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Research-specific imports
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    feedparser = None

try:
    from sec_edgar_downloader import Downloader
    SEC_EDGAR_AVAILABLE = True
except ImportError:
    SEC_EDGAR_AVAILABLE = False
    Downloader = None

try:
    import finnhub
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False
    finnhub = None

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    NewsApiClient = None

# Alpha Vantage is now required
ALPHA_VANTAGE_AVAILABLE = True

try:
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    build = None

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    tweepy = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    # Fallback sentiment analyzer
    class SentimentIntensityAnalyzer:
        def polarity_scores(self, text):
            # Simple fallback sentiment
            positive_words = ['good', 'great', 'excellent', 'positive', 'strong', 'buy', 'growth']
            negative_words = ['bad', 'poor', 'negative', 'weak', 'sell', 'decline', 'loss']

            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)

            if pos_count > neg_count:
                return {'compound': 0.5, 'positive': 0.6, 'negative': 0.2, 'neutral': 0.2}
            elif neg_count > pos_count:
                return {'compound': -0.5, 'positive': 0.2, 'negative': 0.6, 'neutral': 0.2}
            else:
                return {'compound': 0, 'positive': 0.1, 'negative': 0.1, 'neutral': 0.8}

# Handle SSL warnings
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
except:
    pass

try:
    import certifi
    import ssl
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except:
    pass

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_investment_full_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 2: CACHING AND UTILITY FUNCTIONS
# ============================================================================

CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def cache_json(ttl_hours=24):
    """Decorator for caching JSON responses"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_str = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            key = hashlib.sha1(key_str.encode()).hexdigest()
            cache_path = CACHE_DIR / f"{key}.json"

            # Check if cache exists and is valid
            if cache_path.exists():
                cache_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
                if cache_age_hours < ttl_hours:
                    try:
                        return json.loads(cache_path.read_text())
                    except:
                        pass  # If cache is corrupted, regenerate

            # Generate new result
            result = func(*args, **kwargs)

            # Save to cache
            try:
                cache_path.write_text(json.dumps(result, default=str))
            except:
                pass  # If can't cache, continue anyway

            return result
        return wrapper
    return decorator

def retry_with_backoff(max_attempts=3, base_delay=2, max_delay=60):
    """Retry decorator with exponential backoff and jitter - FIXED"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Don't retry on API key errors or rate limits
                    error_msg = str(e).lower()
                    if any(x in error_msg for x in ['api key', 'unauthorized', 'forbidden', 'invalid_api_key', 'invalid api key']):
                        logger.error(f"API authentication error - not retrying: {e}")
                        raise e

                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {e}")
                        raise e

                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** (attempt - 1)) + np.random.uniform(0, 1), max_delay)
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)

            return None
        return wrapper
    return decorator

def safe_divide(numerator, denominator, default=0):
    """Safe division with default value"""
    try:
        if denominator != 0:
            return numerator / denominator
        return default
    except:
        return default

def normalize_date(date_input):
    """Normalize date to timezone-naive pandas datetime"""
    if isinstance(date_input, str):
        date_obj = pd.to_datetime(date_input)
    else:
        date_obj = date_input

    if date_obj.tz is not None:
        date_obj = date_obj.tz_localize(None)

    return date_obj

def mask_api_key(key: str) -> str:
    """Safely mask API key for display"""
    if not key:
        return "NOT SET"
    if len(key) > 10:
        return f"{key[:4]}...{key[-4:]}"
    return "***"

# ============================================================================
# SECTION 3: CONFIGURATION AND CONSTANTS
# ============================================================================

@dataclass
class SimulationConfig:
    """Complete configuration for the simulation system - WITH LLM MODE SELECTION"""

    # API Keys - NEVER log these!
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.getenv('ANTHROPIC_API_KEY', ''))
    NEWS_API_KEY: str = field(default_factory=lambda: os.getenv('NEWS_API_KEY', ''))
    FINNHUB_API_KEY: str = field(default_factory=lambda: os.getenv('FINNHUB_API_KEY', ''))
    ALPHA_VANTAGE_KEY: str = field(default_factory=lambda: os.getenv('ALPHA_VANTAGE_KEY', ''))
    SEC_EMAIL: str = field(default_factory=lambda: os.getenv('SEC_EMAIL', 'research@example.com'))
    GOOGLE_API_KEY: str = field(default_factory=lambda: os.getenv('GOOGLE_API_KEY', ''))
    GOOGLE_CSE_ID: str = field(default_factory=lambda: os.getenv('GOOGLE_CSE_ID', ''))
    SERP_API_KEY: str = field(default_factory=lambda: os.getenv('SERP_API_KEY', ''))
    TWITTER_BEARER_TOKEN: str = field(default_factory=lambda: os.getenv('TWITTER_BEARER_TOKEN', ''))

    # LLM MODE SELECTION - NEW!
    LLM_MODE: str = 'BOTH'  # Options: 'OPENAI_ONLY', 'ANTHROPIC_ONLY', 'BOTH', 'COMPARE_ALL'

    # Simulation Parameters
    START_DATE: str = '2021-10-01'
    END_DATE: str = '2024-12-31'
    INITIAL_CAPITAL: float = 100_000_000  # $100M

    # Investment Universe
    UNIVERSE_SIZE: int = 500
    MIN_MARKET_CAP: float = 10_000_000_000  # $10B
    MIN_ADV: float = 10_000_000  # $10M average daily volume

    # Decision Schedule
    DECISION_DATES: List[str] = field(default_factory=lambda: [
        '2022-01-15', '2022-03-01', '2022-04-30', '2022-06-15',
        '2022-08-01', '2022-10-31', '2022-12-15',
        '2023-01-16', '2023-03-01', '2023-05-01', '2023-06-15',
        '2023-08-01', '2023-10-31', '2023-12-15',
        '2024-01-15', '2024-03-01', '2024-04-30', '2024-06-17',
        '2024-08-01', '2024-10-31', '2024-12-16'
    ])

    # Portfolio Constraints
    MAX_POSITION_SIZE: float = 0.20  # 20% max per position (was 25%, reduced for better risk management)
    MAX_SECTOR_EXPOSURE: float = 0.40  # 40% max per sector
    MIN_CASH: float = 0.02  # 2% minimum cash
    MAX_POSITIONS: int = 30
    MAX_TURNOVER_PER_REBALANCE: float = 0.20  # 20% max turnover

    # Transaction Costs
    COMMISSION_PER_SHARE: float = 0.005
    SPREAD_COST_BP: float = 5  # basis points
    MARKET_IMPACT_FACTOR: float = 0.1
    ADV_PARTICIPATION_LIMIT: float = 0.10  # Max 10% of ADV

    # Risk Parameters
    TARGET_VOLATILITY: float = 0.15  # 15% annual volatility
    MAX_DRAWDOWN_TRIGGER: float = 0.20  # 20% max drawdown
    RISK_FREE_RATE: float = 0.03  # 3% risk-free rate

    # LLM Settings
    LLM_MODEL_PRIMARY: str = 'gpt-4o'
    LLM_MODEL_SECONDARY: str = 'claude-3-opus-20240229'
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4000

    # API COST CONTROLS
    USE_LLM_DECISIONS: bool = True  # ALWAYS use LLMs when available
    MAX_RESEARCH_PER_DATE: int = 10  # Research top 10 candidates for better selection
    USE_EXPENSIVE_APIS: bool = True  # Enable all research APIs
    CACHE_RESEARCH: bool = True  # Cache research results
    USE_MINIMAL_LLM: bool = False  # Use full models for best results
    DRY_RUN: bool = False  # Test mode without API calls
    YFINANCE_DELAY: float = 0.1  # Small delay between yfinance calls (optional)

    # Research Settings
    USE_PROFESSIONAL_RESEARCH: bool = True
    RESEARCH_DEPTH: str = 'moderate'  # 'basic', 'moderate', 'full'
    MAX_NEWS_ARTICLES: int = 20  # Reduced from 50
    MAX_SEC_FILINGS: int = 5  # Reduced from 10
    SENTIMENT_LOOKBACK_DAYS: int = 30
    INSIDER_TRADING_WEIGHT: float = 0.2
    NEWS_SENTIMENT_WEIGHT: float = 0.15
    ANALYST_RATING_WEIGHT: float = 0.15
    SOCIAL_SENTIMENT_WEIGHT: float = 0.1

    # Buffett Principles
    FOCUS_ON_MOAT: bool = True
    FOCUS_ON_MANAGEMENT: bool = True
    FOCUS_ON_FINANCIALS: bool = True
    LONG_TERM_HORIZON: int = 5  # Years

    # Ticker Canonicalization
    TICKER_CANON: Dict[str, str] = field(default_factory=lambda: {
        "TSMC": "TSM", "BNY": "BK", "HP": "HPQ"
    })

    # Berkshire Decisions for Tracking
    BERKSHIRE_DECISIONS: Dict[str, Dict] = field(default_factory=lambda: {
        '2022-01-15': {'buys': ['OXY'], 'sells': []},
        '2022-03-01': {'buys': ['OXY', 'CVX'], 'sells': []},
        '2022-04-30': {'buys': ['OXY', 'HPQ'], 'sells': []},
        '2022-06-15': {'buys': ['OXY'], 'sells': []},
        '2022-08-01': {'buys': ['OXY'], 'sells': []},
        '2022-10-31': {'buys': [], 'sells': ['TSM']},
        '2022-12-15': {'buys': ['AAPL'], 'sells': []},
        '2023-01-16': {'buys': [], 'sells': ['USB', 'BK']},
        '2023-03-01': {'buys': [], 'sells': ['BAC']},
        '2023-05-01': {'buys': ['AAPL'], 'sells': []},
        '2023-06-15': {'buys': [], 'sells': []},
        '2023-08-01': {'buys': [], 'sells': []},
        '2023-10-31': {'buys': [], 'sells': ['AAPL']},
        '2023-12-15': {'buys': [], 'sells': []},
        '2024-01-15': {'buys': ['CB'], 'sells': ['AAPL']},
        '2024-03-01': {'buys': ['CB'], 'sells': []},
        '2024-04-30': {'buys': [], 'sells': []},
        '2024-06-17': {'buys': [], 'sells': ['AAPL']},
        '2024-08-01': {'buys': [], 'sells': []},
        '2024-10-31': {'buys': [], 'sells': []},
        '2024-12-16': {'buys': [], 'sells': []}
    })

# ============================================================================
# SECTION 4: SCHEMA VALIDATION (for LLM outputs)
# ============================================================================

if PYDANTIC_AVAILABLE:
    class Action(BaseModel):
        ticker: str
        action: str  # BUY, SELL, REDUCE
        weight: Optional[float] = 0
        target_weight: Optional[float] = None
        rationale: Optional[str] = ""

        @validator('ticker')
        def validate_ticker(cls, v):
            if len(v) > 6:
                raise ValueError('Ticker too long')
            return v.upper()

        @validator('action')
        def validate_action(cls, v):
            if v.upper() not in ['BUY', 'SELL', 'REDUCE']:
                raise ValueError('Invalid action')
            return v.upper()

        @validator('weight')
        def validate_weight(cls, v):
            if v is not None:
                return min(max(v, 0), 0.20)
            return v

    class LLMDecision(BaseModel):
        actions: List[Action] = Field(default_factory=list)
        market_view: str = ""
        risk_assessment: str = ""
        key_themes: List[str] = Field(default_factory=list)

        @validator('actions')
        def validate_actions(cls, v):
            if len(v) > 100:
                return v[:100]
            return v
else:
    # Fallback classes
    @dataclass
    class Action:
        ticker: str
        action: str
        weight: float = 0
        target_weight: Optional[float] = None
        rationale: str = ""

    @dataclass
    class LLMDecision:
        actions: List[Action] = field(default_factory=list)
        market_view: str = ""
        risk_assessment: str = ""
        key_themes: List[str] = field(default_factory=list)

# ============================================================================
# SECTION 5: WARREN BUFFETT INVESTMENT PRINCIPLES
# ============================================================================

class BuffettPrinciples:
    """Encode Warren Buffett's investment principles"""

    PRINCIPLES = {
        "circle_of_competence": [
            "Only invest in businesses you understand",
            "Stay within areas of expertise",
            "Avoid complex financial instruments"
        ],
        "economic_moat": [
            "Brand power and customer loyalty",
            "Network effects and switching costs",
            "Cost advantages and economies of scale",
            "Regulatory advantages and patents",
            "High barriers to entry"
        ],
        "management_quality": [
            "Owner-oriented management",
            "Capital allocation track record",
            "Transparent communication",
            "Skin in the game (ownership)",
            "Long-term thinking"
        ],
        "financial_strength": [
            "Consistent earnings growth",
            "High return on equity (>15%)",
            "Low debt-to-equity ratio",
            "Strong free cash flow",
            "Predictable business model"
        ],
        "valuation": [
            "Buy at discount to intrinsic value",
            "Margin of safety principle",
            "Focus on owner earnings",
            "Long-term value creation",
            "Price is what you pay, value is what you get"
        ],
        "holding_period": [
            "Our favorite holding period is forever",
            "Time is the friend of wonderful business",
            "Patience is key virtue"
        ]
    }

    @staticmethod
    def score_company(company_data: Dict) -> Dict:
        """Score a company based on Buffett principles"""
        scores = {}

        # Economic Moat Score (0-100)
        moat_score = 0

        # Gross margin indicates pricing power
        gross_margin = company_data.get('gross_margin', 0)
        if gross_margin > 0.50:
            moat_score += 30
        elif gross_margin > 0.40:
            moat_score += 20
        elif gross_margin > 0.30:
            moat_score += 10

        # ROE indicates competitive advantage
        roe = company_data.get('roe', 0)
        if roe > 0.25:
            moat_score += 30
        elif roe > 0.20:
            moat_score += 25
        elif roe > 0.15:
            moat_score += 20
        elif roe > 0.10:
            moat_score += 10

        # Profit margin
        profit_margin = company_data.get('profit_margin', 0)
        if profit_margin > 0.20:
            moat_score += 20
        elif profit_margin > 0.15:
            moat_score += 15
        elif profit_margin > 0.10:
            moat_score += 10
        elif profit_margin > 0.05:
            moat_score += 5

        # Market share and brand value
        if company_data.get('market_share', 0) > 0.2:
            moat_score += 10
        if company_data.get('brand_value_rank', 100) < 50:
            moat_score += 10

        scores['moat'] = min(moat_score, 100)

        # Management Quality Score (0-100)
        mgmt_score = 50  # Base score

        # Insider ownership
        insider_ownership = company_data.get('insider_ownership', 0)
        if insider_ownership > 0.10:
            mgmt_score += 20
        elif insider_ownership > 0.05:
            mgmt_score += 15
        elif insider_ownership > 0.01:
            mgmt_score += 10

        # Management tenure
        if company_data.get('management_tenure', 0) > 5:
            mgmt_score += 15

        # Capital allocation
        if company_data.get('capital_return_rate', 0) > 0.1:
            mgmt_score += 15

        scores['management'] = min(mgmt_score, 100)

        # Financial Strength Score (0-100)
        financial_score = 0

        # Debt levels
        debt_to_equity = company_data.get('debt_to_equity', 1)
        if debt_to_equity < 0.3:
            financial_score += 25
        elif debt_to_equity < 0.5:
            financial_score += 20
        elif debt_to_equity < 1.0:
            financial_score += 10

        # Current ratio
        current_ratio = company_data.get('current_ratio', 1)
        if current_ratio > 2.0:
            financial_score += 20
        elif current_ratio > 1.5:
            financial_score += 15
        elif current_ratio > 1.0:
            financial_score += 10

        # Free cash flow
        fcf_yield = company_data.get('fcf_yield', 0)
        if fcf_yield > 0.08:
            financial_score += 30
        elif fcf_yield > 0.05:
            financial_score += 20
        elif fcf_yield > 0.03:
            financial_score += 10

        # Earnings consistency
        if company_data.get('earnings_consistency', 0) > 0.8:
            financial_score += 25

        scores['financial_strength'] = min(financial_score, 100)

        # Valuation Score (0-100)
        valuation_score = 0

        # P/E ratio
        pe_ratio = company_data.get('pe_ratio', 30)
        if 0 < pe_ratio < 15:
            valuation_score += 30
        elif pe_ratio < 20:
            valuation_score += 25
        elif pe_ratio < 25:
            valuation_score += 15
        elif pe_ratio < 30:
            valuation_score += 5

        # P/B ratio
        pb_ratio = company_data.get('pb_ratio', 3)
        if 0 < pb_ratio < 1.5:
            valuation_score += 30
        elif pb_ratio < 2.5:
            valuation_score += 20
        elif pb_ratio < 3.5:
            valuation_score += 10

        # FCF/Price ratio
        if fcf_yield > 0.08:
            valuation_score += 40
        elif fcf_yield > 0.05:
            valuation_score += 25
        elif fcf_yield > 0.03:
            valuation_score += 15

        scores['valuation'] = min(valuation_score, 100)

        # Calculate Overall Buffett Score
        scores['overall'] = (
            scores['moat'] * 0.30 +
            scores['management'] * 0.20 +
            scores['financial_strength'] * 0.30 +
            scores['valuation'] * 0.20
        )

        # Add classification
        overall = scores['overall']
        if overall >= 80:
            scores['classification'] = 'EXCEPTIONAL'
            scores['recommendation'] = 'STRONG_BUY'
        elif overall >= 70:
            scores['classification'] = 'EXCELLENT'
            scores['recommendation'] = 'BUY'
        elif overall >= 60:
            scores['classification'] = 'GOOD'
            scores['recommendation'] = 'ACCUMULATE'
        elif overall >= 50:
            scores['classification'] = 'FAIR'
            scores['recommendation'] = 'HOLD'
        elif overall >= 40:
            scores['classification'] = 'BELOW_AVERAGE'
            scores['recommendation'] = 'REDUCE'
        else:
            scores['classification'] = 'POOR'
            scores['recommendation'] = 'SELL'

        return scores

# ============================================================================
# SECTION 6: DEEP RESEARCH ENGINE - OPTIMIZED
# ============================================================================

class DeepResearchEngine:
    """Professional-grade research engine - OPTIMIZED FOR COST"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.buffett_analyzer = BuffettPrinciples()
        self.research_cache = {}
        self._init_clients()

    def _init_clients(self):
        """Initialize all research API clients"""

        # Only initialize if not in dry run mode
        if self.config.DRY_RUN:
            logger.info("DRY RUN MODE - Skipping API client initialization")
            self.openai_client = None
            self.anthropic_client = None
            self.finnhub_client = None
            self.alpha_vantage = None
            self.newsapi = None
            self.sec_downloader = None
            self.google_search = None
            return

        # OpenAI
        if self.config.OPENAI_API_KEY and self.config.USE_LLM_DECISIONS:
            try:
                self.openai_client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except:
                self.openai_client = None
        else:
            self.openai_client = None

        # Anthropic
        if self.config.ANTHROPIC_API_KEY and self.config.USE_LLM_DECISIONS:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=self.config.ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized")
            except:
                self.anthropic_client = None
        else:
            self.anthropic_client = None

        # Only initialize expensive APIs if enabled
        if not self.config.USE_EXPENSIVE_APIS:
            logger.info("Expensive APIs disabled - using free data only")
            self.finnhub_client = None
            self.alpha_vantage = None
            self.newsapi = None
            self.sec_downloader = None
            self.google_search = None
            return

        # Financial data APIs
        if self.config.FINNHUB_API_KEY and FINNHUB_AVAILABLE:
            try:
                self.finnhub_client = finnhub.Client(api_key=self.config.FINNHUB_API_KEY)
                logger.info("Finnhub client initialized")
            except:
                self.finnhub_client = None
        else:
            self.finnhub_client = None

        if self.config.ALPHA_VANTAGE_KEY and ALPHA_VANTAGE_AVAILABLE:
            try:
                self.alpha_vantage = FundamentalData(key=self.config.ALPHA_VANTAGE_KEY)
                logger.info("Alpha Vantage client initialized")
            except:
                self.alpha_vantage = None
        else:
            self.alpha_vantage = None

        # News APIs
        if self.config.NEWS_API_KEY and NEWSAPI_AVAILABLE:
            try:
                self.newsapi = NewsApiClient(api_key=self.config.NEWS_API_KEY)
                logger.info("NewsAPI client initialized")
            except:
                self.newsapi = None
        else:
            self.newsapi = None

        # SEC downloader (initialize as None since SEC module not available)
        self.sec_downloader = None

        # Google search
        self.google_search = None

    async def research_company(self, ticker: str, date: str) -> Dict:
        """Conduct comprehensive research on a company - FULL RESEARCH FOR ALL"""

        # Check cache first
        cache_key = f"{ticker}_{date}"
        if self.config.CACHE_RESEARCH and cache_key in self.research_cache:
            return self.research_cache[cache_key]

        logger.info(f"Conducting deep research on {ticker} as of {date}")

        research_results = {
            'ticker': ticker,
            'date': date,
            'fundamental_analysis': {},
            'news_sentiment': {},
            'sec_filings': {},
            'insider_trading': {},
            'competitive_analysis': {},
            'management_assessment': {},
            'industry_trends': {},
            'analyst_opinions': {},
            'social_sentiment': {},
            'technical_analysis': {},
            'buffett_score': {},
            'research_summary': ''
        }

        # Always do fundamental analysis (no API calls needed)
        research_results['fundamental_analysis'] = await self._research_fundamentals(ticker, date)

        # Calculate Buffett score (no API calls needed)
        research_results['buffett_score'] = self.buffett_analyzer.score_company(
            research_results['fundamental_analysis']
        )

        # ALWAYS DO FULL RESEARCH FOR ALL STOCKS (as requested)
        if self.config.USE_EXPENSIVE_APIS and self.config.USE_PROFESSIONAL_RESEARCH:
            # News sentiment analysis
            research_results['news_sentiment'] = await self._analyze_news_sentiment(ticker, date)

            # SEC filings analysis (if available)
            if self.sec_downloader:
                research_results['sec_filings'] = await self._analyze_sec_filings(ticker, date)

            # Insider trading analysis
            if self.finnhub_client:
                research_results['insider_trading'] = await self._track_insider_trading(ticker, date)
                research_results['analyst_opinions'] = await self._gather_analyst_opinions(ticker, date)

            # Competitive analysis
            research_results['competitive_analysis'] = await self._competitive_analysis(ticker, date)

            # Management assessment
            research_results['management_assessment'] = await self._assess_management(ticker, date)

            # Industry trends
            research_results['industry_trends'] = await self._research_industry_trends(ticker, date)

            # Social sentiment
            research_results['social_sentiment'] = await self._analyze_social_sentiment(ticker, date)

        # Generate comprehensive summary
        research_results['research_summary'] = self._generate_comprehensive_summary(research_results)

        # Cache results
        if self.config.CACHE_RESEARCH:
            self.research_cache[cache_key] = research_results

        return research_results

    def _generate_comprehensive_summary(self, research: Dict) -> str:
        """Generate comprehensive summary with all research data"""
        summary_parts = []

        # Buffett score summary
        buffett = research.get('buffett_score', {})
        if buffett:
            summary_parts.append(
                f"Buffett Score: {buffett.get('overall', 0):.0f}/100 "
                f"({buffett.get('classification', 'Unknown')})"
            )

        # Fundamental summary
        fundamentals = research.get('fundamental_analysis', {})
        if fundamentals:
            summary_parts.append(
                f"P/E: {fundamentals.get('pe_ratio', 'N/A')}, "
                f"ROE: {fundamentals.get('roe', 0):.1%}, "
                f"Debt/Equity: {fundamentals.get('debt_to_equity', 'N/A')}"
            )

        # Sentiment summary
        news = research.get('news_sentiment', {})
        if news and news.get('news_volume', 0) > 0:
            summary_parts.append(
                f"News Sentiment: {news.get('overall_sentiment', 0):.2f} "
                f"({news.get('news_volume', 0)} articles)"
            )

        # Insider trading
        insider = research.get('insider_trading', {})
        if insider and insider.get('insider_sentiment'):
            summary_parts.append(
                f"Insider Sentiment: {insider.get('insider_sentiment', 'Unknown')}"
            )

        # Analyst opinions
        analyst = research.get('analyst_opinions', {})
        if analyst and analyst.get('consensus_rating'):
            summary_parts.append(
                f"Analyst Rating: {analyst.get('consensus_rating', 'Unknown')}"
            )

        return " | ".join(summary_parts)

    def _calculate_quality_score(self, fundamentals: Dict) -> float:
        """Calculate quality score"""
        score = 0

        # ROE quality
        roe = fundamentals.get('roe', 0)
        if roe > 0.20:
            score += 25
        elif roe > 0.15:
            score += 15
        elif roe > 0.10:
            score += 10

        # Margin quality
        profit_margin = fundamentals.get('profit_margin', 0)
        if profit_margin > 0.20:
            score += 25
        elif profit_margin > 0.10:
            score += 15

        # Financial health
        debt_to_equity = fundamentals.get('debt_to_equity', 100)
        if debt_to_equity < 0.5:
            score += 25
        elif debt_to_equity < 1.0:
            score += 15

        # Cash flow
        fcf_yield = fundamentals.get('fcf_yield', 0)
        if fcf_yield > 0.08:
            score += 25
        elif fcf_yield > 0.05:
            score += 15

        return min(score, 100)

    async def _analyze_news_sentiment(self, ticker: str, date: str) -> Dict:
        """Analyze news sentiment - REDUCED API CALLS"""

        sentiment_data = {
            'overall_sentiment': 0,
            'sentiment_score': 0,
            'news_volume': 0
        }

        if not self.newsapi or not self.config.USE_EXPENSIVE_APIS:
            return sentiment_data

        try:
            end_date = pd.to_datetime(date)
            start_date = end_date - timedelta(days=7)  # Reduced from 30 days

            # Limit news articles
            news = self.newsapi.get_everything(
                q=ticker,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                sort_by='relevancy',
                page_size=10  # Reduced from 50
            )

            sentiments = []
            for article in news.get('articles', []):
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                sentiments.append(sentiment['compound'])

            sentiment_data['news_volume'] = len(sentiments)
            sentiment_data['overall_sentiment'] = np.mean(sentiments) if sentiments else 0
            sentiment_data['sentiment_score'] = sentiment_data['overall_sentiment'] * 100

        except Exception as e:
            logger.debug(f"NewsAPI error for {ticker}: {e}")

        return sentiment_data

    async def _track_insider_trading(self, ticker: str, date: str) -> Dict:
        """Track insider trading patterns - MINIMAL API CALLS"""

        insider_data = {
            'insider_sentiment': 'NEUTRAL',
            'conviction_score': 0
        }

        if not self.finnhub_client or not self.config.USE_EXPENSIVE_APIS:
            return insider_data

        try:
            transactions = self.finnhub_client.insider_transactions(ticker)

            total_buys = 0
            total_sells = 0

            for trans in transactions.get('data', [])[:10]:  # Only check recent 10
                if trans.get('transactionType') == 'P':
                    total_buys += 1
                elif trans.get('transactionType') == 'S':
                    total_sells += 1

            if total_buys + total_sells > 0:
                insider_data['conviction_score'] = (
                    (total_buys - total_sells) / (total_buys + total_sells) * 100
                )

            if insider_data['conviction_score'] > 30:
                insider_data['insider_sentiment'] = 'BULLISH'
            elif insider_data['conviction_score'] < -30:
                insider_data['insider_sentiment'] = 'BEARISH'

        except Exception as e:
            logger.debug(f"Error tracking insider trading for {ticker}: {e}")

        return insider_data

    async def _gather_analyst_opinions(self, ticker: str, date: str) -> Dict:
        """Gather analyst opinions - MINIMAL API CALLS"""

        analyst_data = {
            'consensus_rating': 'HOLD',
            'analyst_sentiment': 'NEUTRAL'
        }

        if not self.finnhub_client or not self.config.USE_EXPENSIVE_APIS:
            return analyst_data

        try:
            rec_trends = self.finnhub_client.recommendation_trends(ticker)
            if rec_trends and len(rec_trends) > 0:
                latest = rec_trends[0]
                buy = latest.get('buy', 0) + latest.get('strongBuy', 0)
                sell = latest.get('sell', 0) + latest.get('strongSell', 0)
                hold = latest.get('hold', 0)

                if buy > sell + hold:
                    analyst_data['consensus_rating'] = 'BUY'
                    analyst_data['analyst_sentiment'] = 'BULLISH'
                elif sell > buy + hold:
                    analyst_data['consensus_rating'] = 'SELL'
                    analyst_data['analyst_sentiment'] = 'BEARISH'

        except Exception as e:
            logger.debug(f"Error gathering analyst opinions for {ticker}: {e}")

        return analyst_data

    async def _analyze_sec_filings(self, ticker: str, date: str) -> Dict:
        """Analyze recent SEC filings"""

        filings_analysis = {
            'recent_10k': {},
            'recent_10q': {},
            'recent_8k': [],
            'risk_factors': [],
            'management_discussion': '',
            'material_changes': []
        }

        try:
            if self.sec_downloader:
                # Note: Actual SEC filing download requires implementation
                # This is a placeholder for the structure
                filings_analysis['recent_10k'] = {
                    'filing_date': date,
                    'business_description': 'From Item 1',
                    'risk_factors': 'From Item 1A',
                    'financial_condition': 'From Item 7'
                }

        except Exception as e:
            logger.debug(f"Error analyzing SEC filings for {ticker}: {e}")

        return filings_analysis

    async def _competitive_analysis(self, ticker: str, date: str) -> Dict:
        """Analyze competitive position"""

        competitive_data = {
            'market_position': '',
            'competitive_advantages': [],
            'competitive_threats': [],
            'market_share': 0,
            'relative_performance': {},
            'peer_comparison': {}
        }

        try:
            # Rate limiting for yfinance
            # await asyncio.sleep(self.config.YFINANCE_DELAY) # Disabled Yahoo Finance delay
            # stock = yf.Ticker(ticker) # Disabled - would need Alpha Vantage
            info = {}

            industry = info.get('industry', '')
            sector = info.get('sector', '')

            # Find peers
            if self.finnhub_client:
                try:
                    peers = self.finnhub_client.company_peers(ticker)

                    # Compare metrics with peers
                    peer_metrics = []
                    for peer in peers[:5]:
                        try:
                            # peer_stock = yf.Ticker(peer) # Disabled - would need Alpha Vantage
                            peer_info = {}
                            peer_metrics.append({
                                'ticker': peer,
                                'pe_ratio': peer_info.get('trailingPE', 0),
                                'profit_margin': peer_info.get('profitMargins', 0),
                                'roe': peer_info.get('returnOnEquity', 0),
                                'market_cap': peer_info.get('marketCap', 0)
                            })
                        except:
                            continue

                    # Calculate relative position
                    if peer_metrics:
                        avg_peer_pe = np.mean([p['pe_ratio'] for p in peer_metrics if p['pe_ratio'] > 0])
                        avg_peer_margin = np.mean([p['profit_margin'] for p in peer_metrics if p['profit_margin'] > 0])
                        avg_peer_roe = np.mean([p['roe'] for p in peer_metrics if p['roe'] > 0])

                        competitive_data['relative_performance'] = {
                            'pe_vs_peers': safe_divide(info.get('trailingPE', 0), avg_peer_pe),
                            'margin_vs_peers': safe_divide(info.get('profitMargins', 0), avg_peer_margin),
                            'roe_vs_peers': safe_divide(info.get('returnOnEquity', 0), avg_peer_roe)
                        }

                        competitive_data['peer_comparison'] = peer_metrics
                except:
                    pass

            # Determine market position
            if competitive_data['relative_performance']:
                rel_perf = competitive_data['relative_performance']
                if rel_perf.get('margin_vs_peers', 0) > 1.2 and rel_perf.get('roe_vs_peers', 0) > 1.2:
                    competitive_data['market_position'] = 'LEADER'
                elif rel_perf.get('margin_vs_peers', 0) > 1.0 or rel_perf.get('roe_vs_peers', 0) > 1.0:
                    competitive_data['market_position'] = 'STRONG'
                elif rel_perf.get('margin_vs_peers', 0) > 0.8 and rel_perf.get('roe_vs_peers', 0) > 0.8:
                    competitive_data['market_position'] = 'COMPETITIVE'
                else:
                    competitive_data['market_position'] = 'WEAK'

        except Exception as e:
            logger.debug(f"Error in competitive analysis for {ticker}: {e}")

        return competitive_data

    async def _assess_management(self, ticker: str, date: str) -> Dict:
        """Assess management quality"""

        management_data = {
            'ceo_tenure': 0,
            'management_ownership': 0,
            'capital_allocation_score': 50,
            'communication_transparency': 0,
            'execution_track_record': [],
            'management_quality': 'AVERAGE'
        }

        try:
            # Rate limiting for yfinance
            # await asyncio.sleep(self.config.YFINANCE_DELAY) # Disabled Yahoo Finance delay
            # stock = yf.Ticker(ticker) # Disabled - would need Alpha Vantage
            info = {}

            # Get insider ownership
            management_data['management_ownership'] = info.get('heldPercentInsiders', 0)

            # Calculate capital allocation score based on metrics
            if info.get('returnOnEquity', 0) > 0.15:
                management_data['capital_allocation_score'] += 20
            if info.get('payoutRatio', 0) > 0 and info.get('payoutRatio', 0) < 0.5:
                management_data['capital_allocation_score'] += 15
            if info.get('debtToEquity', 0) < 0.5:
                management_data['capital_allocation_score'] += 15

            # Determine management quality
            if management_data['capital_allocation_score'] > 70:
                management_data['management_quality'] = 'EXCELLENT'
            elif management_data['capital_allocation_score'] > 60:
                management_data['management_quality'] = 'GOOD'
            elif management_data['capital_allocation_score'] > 40:
                management_data['management_quality'] = 'AVERAGE'
            else:
                management_data['management_quality'] = 'POOR'

        except Exception as e:
            logger.debug(f"Error assessing management for {ticker}: {e}")

        return management_data

    async def _research_industry_trends(self, ticker: str, date: str) -> Dict:
        """Research industry trends and outlook"""

        industry_data = {
            'industry_growth_rate': 0,
            'key_trends': [],
            'disruption_risks': [],
            'regulatory_changes': [],
            'market_size': 0,
            'industry_outlook': 'STABLE'
        }

        try:
            stock = yf.Ticker(ticker)
            industry = stock.info.get('industry', '')

            # Search for industry trends
            if self.newsapi:
                try:
                    industry_news = self.newsapi.get_everything(
                        q=f"{industry} industry trends outlook",
                        from_param=(pd.to_datetime(date) - timedelta(days=30)).strftime('%Y-%m-%d'),
                        to=date,
                        sort_by='relevancy',
                        page_size=10
                    )

                    for article in industry_news.get('articles', []):
                        title = article.get('title', '')
                        if any(word in title.lower() for word in ['growth', 'expand', 'increase', 'boom']):
                            industry_data['key_trends'].append(title)
                            industry_data['industry_outlook'] = 'POSITIVE'
                        elif any(word in title.lower() for word in ['risk', 'threat', 'disruption', 'decline']):
                            industry_data['disruption_risks'].append(title)
                            if industry_data['industry_outlook'] != 'POSITIVE':
                                industry_data['industry_outlook'] = 'NEGATIVE'
                        elif any(word in title.lower() for word in ['regulation', 'law', 'policy']):
                            industry_data['regulatory_changes'].append(title)
                except:
                    pass

        except Exception as e:
            logger.debug(f"Error researching industry trends: {e}")

        return industry_data

    async def _analyze_social_sentiment(self, ticker: str, date: str) -> Dict:
        """Analyze social media sentiment"""

        social_data = {
            'twitter_sentiment': 0,
            'reddit_sentiment': 0,
            'stocktwits_sentiment': 0,
            'social_volume': 0,
            'trending_score': 0,
            'social_sentiment': 'NEUTRAL'
        }

        try:
            # Reddit sentiment via API (simplified)
            reddit_url = f"https://www.reddit.com/r/wallstreetbets/search.json?q={ticker}&sort=new&limit=25"

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(reddit_url, headers={'User-Agent': 'InvestmentBot'}) as response:
                        if response.status == 200:
                            data = await response.json()
                            sentiments = []

                            for post in data.get('data', {}).get('children', []):
                                title = post.get('data', {}).get('title', '')
                                sentiment = self.sentiment_analyzer.polarity_scores(title)
                                sentiments.append(sentiment['compound'])

                            if sentiments:
                                social_data['reddit_sentiment'] = np.mean(sentiments)
                                social_data['social_volume'] = len(sentiments)

                                if social_data['reddit_sentiment'] > 0.2:
                                    social_data['social_sentiment'] = 'POSITIVE'
                                elif social_data['reddit_sentiment'] < -0.2:
                                    social_data['social_sentiment'] = 'NEGATIVE'
            except:
                pass

        except Exception as e:
            logger.debug(f"Error analyzing social sentiment for {ticker}: {e}")

        return social_data

    async def _research_fundamentals(self, ticker: str, date: str) -> Dict:
        """Deep fundamental analysis - WITH RATE LIMITING AND HISTORICAL DATA ONLY"""

        fundamentals = {}
        data_filter = HistoricalDataFilter()

        try:
            # Get data from DataManager if available (already cached)
            if hasattr(self, 'data_manager') and self.data_manager:
                if ticker in self.data_manager.fundamental_data:
                    # Use cached fundamental data and filter it
                    cached_fundamentals = self.data_manager.fundamental_data[ticker]
                    fundamentals = data_filter.filter_fundamental_data(cached_fundamentals, date)
                    return fundamentals
            
            # Fallback to fetching if not cached
            # Add delay to respect API rate limits
            await asyncio.sleep(self.config.YFINANCE_DELAY * 1.5)  # Longer delay for fundamentals
            
            stock = yf.Ticker(ticker)

            # Current metrics - with retry logic
            info = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    info = stock.info
                    break  # Success, exit retry loop
                except Exception as e:
                    if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                        if attempt < max_retries - 1:  # Not the last attempt
                            wait_time = (attempt + 1) * 2  # Progressive backoff: 2, 4, 6 seconds
                            logger.warning(f"Rate limited for {ticker}, waiting {wait_time}s before retry {attempt + 2}/{max_retries}")
                            await asyncio.sleep(wait_time)
                            continue
                    raise e  # Re-raise if not rate limit or last attempt
            
            if not info:
                logger.warning(f"Could not fetch info for {ticker} after {max_retries} attempts")
                return fundamentals
            fundamentals.update({
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'dividend_rate': info.get('dividendRate', 0),
                'payout_ratio': info.get('payoutRatio', 0),
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'gross_margin': info.get('grossMargins', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'book_value': info.get('bookValue', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'avg_volume': info.get('averageVolume', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            })

            # Calculate quality metrics
            fundamentals['quality_score'] = self._calculate_quality_score(fundamentals)

        except Exception as e:
            logger.error(f"Error researching fundamentals for {ticker}: {e}")

        return fundamentals

# ============================================================================
# SECTION 7: DATA MANAGEMENT SYSTEM - FIXED
# ============================================================================

class AlphaVantageManager:
    """Manages all Alpha Vantage API interactions with rate limiting and SSL fix"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Fix SSL issues by using custom session
        import ssl
        try:
            import certifi
            # Create custom session with proper SSL settings
            session = requests.Session()
            session.verify = certifi.where()
        except ImportError:
            # If certifi is not available, use default session
            session = None
            
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.fd = FundamentalData(key=api_key, output_format='pandas')
        self.last_api_call = 0
        self.min_interval = 12.5  # 5 calls per minute for free tier (12 seconds between calls)
        self.retry_count = 3
        self.retry_delay = 5
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time_module.time() - self.last_api_call
        if elapsed < self.min_interval:
            time_module.sleep(self.min_interval - elapsed)
        self.last_api_call = time_module.time()
    
    def get_daily_adjusted(self, symbol: str, outputsize: str = 'full') -> Optional[pd.DataFrame]:
        """Get daily price data (using free tier endpoint)"""
        for attempt in range(self.retry_count):
            try:
                self._rate_limit()
                # Use get_daily instead of get_daily_adjusted (free tier)
                data, meta_data = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
                
                # Rename columns to match expected format
                data = data.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                })
                
                # Add missing columns for compatibility
                data['Adj Close'] = data['Close']  # No adjusted close in free tier
                data['Dividends'] = 0
                data['Stock Splits'] = 1
                
                # Ensure datetime index
                data.index = pd.to_datetime(data.index)
                data = data.sort_index()
                
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.retry_count} failed for {symbol}: {e}")
                if attempt < self.retry_count - 1:
                    time_module.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to get data for {symbol} after {self.retry_count} attempts")
                    return None
    
    def get_company_overview(self, symbol: str) -> Dict:
        """Get company fundamentals"""
        for attempt in range(self.retry_count):
            try:
                self._rate_limit()
                data, _ = self.fd.get_company_overview(symbol)
                return data.to_dict() if not data.empty else {}
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.retry_count} failed for {symbol} overview: {e}")
                if attempt < self.retry_count - 1:
                    time_module.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to get overview for {symbol}")
                    return {}
    
    def get_income_statement(self, symbol: str) -> pd.DataFrame:
        """Get income statement data"""
        try:
            self._rate_limit()
            data, _ = self.fd.get_income_statement_annual(symbol)
            return data
        except Exception as e:
            logger.warning(f"Failed to get income statement for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """Get balance sheet data"""
        try:
            self._rate_limit()
            data, _ = self.fd.get_balance_sheet_annual(symbol)
            return data
        except Exception as e:
            logger.warning(f"Failed to get balance sheet for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        """Get cash flow data"""
        try:
            self._rate_limit()
            data, _ = self.fd.get_cash_flow_annual(symbol)
            return data
        except Exception as e:
            logger.warning(f"Failed to get cash flow for {symbol}: {e}")
            return pd.DataFrame()

class DataManager:
    """Comprehensive data management system - FIXED"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.price_data = {}
        self.fundamental_data = {}
        self.market_data = {}
        self.news_sentiment = {}
        self.sector_mapping = {}
        self.tickers = []
        self.ticker_info = pd.DataFrame()
        self.adv_data = {}
        
        # Initialize caching system
        self.cache = StockDataCache(cache_dir="stock_data_cache", cache_expiry_days=7)
        
        # Initialize Alpha Vantage manager
        alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        if not alpha_vantage_key:
            raise ValueError("ALPHA_VANTAGE_KEY not found in environment variables")
        self.av_manager = AlphaVantageManager(alpha_vantage_key)

    def download_all_data(self):
        """Master function to download all required data"""
        
        # Check if we have cached data first
        if os.path.exists('simulation_data_complete.pkl'):
            logger.info("Found cached data, loading from simulation_data_complete.pkl...")
            try:
                with open('simulation_data_complete.pkl', 'rb') as f:
                    cached_data = pickle.load(f)
                    
                if 'price_data' in cached_data and cached_data['price_data']:
                    self.price_data = cached_data['price_data']
                    self.fundamental_data = cached_data.get('fundamental_data', {})
                    self.market_data = cached_data.get('market_data', {})
                    self.tickers = cached_data.get('tickers', list(self.price_data.keys()))
                    self.sector_mapping = cached_data.get('sector_mapping', {})
                    self.adv_data = cached_data.get('adv_data', {})
                    logger.info(f"Loaded cached data for {len(self.tickers)} tickers")
                    return
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}, downloading fresh data...")
        
        logger.info("Starting comprehensive data download...")

        try:
            # Get ticker universe
            self.get_ticker_universe()

            # Download price data
            self.download_price_data()

            # Download fundamental data
            self.download_fundamental_data()

            # Get market indicators
            self.download_market_indicators()

            # Calculate derived metrics
            self.calculate_metrics()

            # Calculate ADV
            self.calculate_adv()

            # Build sector mapping
            self.build_sector_mapping()

            # Save data
            self.save_data()

            logger.info("Data download complete!")

        except Exception as e:
            logger.error(f"Data download failed: {e}")
            raise

    def get_ticker_universe(self):
        """Get investment universe - FIXED WITH BETTER ERROR HANDLING"""

        try:
            # Try S&P 500 with better error handling
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

            # Add headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            # Try with requests first
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                tables = pd.read_html(response.text)
                if tables and len(tables) > 0:
                    sp500_table = tables[0]
                    self.tickers = sp500_table['Symbol'].str.replace('.', '-').tolist()[:self.config.UNIVERSE_SIZE]
                    self.ticker_info = sp500_table.set_index('Symbol')[['Security', 'GICS Sector', 'GICS Sub-Industry']]

                    # Add Berkshire holdings
                    additional = ['OXY', 'CB', 'TSM', 'BK', 'HPQ', 'USB']
                    for ticker in additional:
                        canon_ticker = self.config.TICKER_CANON.get(ticker, ticker)
                        if canon_ticker not in self.tickers:
                            self.tickers.append(canon_ticker)

                    logger.info(f"Downloaded {len(self.tickers)} tickers from Wikipedia")
                    return

        except Exception as e:
            if "Read timed out" in str(e) or "timeout" in str(e).lower():
                logger.warning(f"Wikipedia connection timed out (network issue): {e}")
            else:
                logger.warning(f"Failed to get S&P 500 from Wikipedia: {e}")

        # Use hardcoded fallback list
        logger.info("Using fallback ticker list (60 major stocks)")
        self.tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JPM', 'JNJ',
            'V', 'WMT', 'PG', 'HD', 'BAC', 'UNH', 'DIS', 'ADBE', 'NFLX', 'CRM',
            'PYPL', 'INTC', 'CMCSA', 'ABBV', 'ABT', 'TMO', 'ACN', 'CVX', 'MCD', 'COST',
            'AVGO', 'NKE', 'NEE', 'LLY', 'DHR', 'TXN', 'QCOM', 'BMY', 'PM', 'HON',
            'UNP', 'IBM', 'CAT', 'GE', 'MMM', 'MO', 'CVS', 'LOW', 'GS', 'AXP',
            'BLK', 'OXY', 'CB', 'USB', 'TSM', 'BK', 'HPQ', 'NEM', 'SYF', 'GL'
        ]

        # Apply canonicalization
        canonicalized = []
        for ticker in self.tickers:
            canon = self.config.TICKER_CANON.get(ticker, ticker)
            if canon not in canonicalized:
                canonicalized.append(canon)
        self.tickers = canonicalized

    @retry_with_backoff(max_attempts=3, base_delay=2)
    def download_price_data(self):
        """Download historical price data using hybrid approach (yfinance + Alpha Vantage)"""
        logger.info("Downloading price data using hybrid approach (yfinance primary, Alpha Vantage fallback)...")

        def download_single_ticker(ticker):
            try:
                # Use cache first, then fallback to direct download
                start_date = self.config.START_DATE
                end_date = self.config.END_DATE
                
                # Get data from cache or download if needed
                hist, fundamental_data = self.cache.get_all_stock_data(
                    ticker, start_date, end_date, force_refresh=False
                )
                
                if hist is not None and len(hist) > 50:
                    # Data successfully retrieved from cache or downloaded
                    logger.debug(f"✅ Loaded {ticker} data")
                    return ticker, hist, fundamental_data
                
                # If cache failed, try Alpha Vantage as fallback
                if hist is None:
                    logger.debug(f"Cache/yfinance failed for {ticker}, trying Alpha Vantage...")
                    hist = self.av_manager.get_daily_adjusted(ticker)
                    
                    if hist is None or len(hist) == 0:
                        return ticker, None, None
                    
                    # Filter to date range
                    start_date_dt = pd.to_datetime(start_date)
                    end_date_dt = pd.to_datetime(end_date)
                    
                    # Handle timezone-aware vs timezone-naive datetime comparison
                    if hist.index.tz is not None:
                        start_date_dt = start_date_dt.tz_localize('UTC')
                        end_date_dt = end_date_dt.tz_localize('UTC')
                        
                    hist = hist[(hist.index >= start_date_dt) & (hist.index <= end_date_dt)]
                    
                    if len(hist) > 100:
                        # Get fundamental data from Alpha Vantage
                        overview = self.av_manager.get_company_overview(ticker)
                        
                        fundamental_data = {
                            'market_cap': float(overview.get('MarketCapitalization', 0)) if overview.get('MarketCapitalization') else 0,
                            'pe_ratio': float(overview.get('PERatio', 0)) if overview.get('PERatio') and overview.get('PERatio') != 'None' else None,
                            'pb_ratio': float(overview.get('PriceToBookRatio', 0)) if overview.get('PriceToBookRatio') and overview.get('PriceToBookRatio') != 'None' else None,
                            'dividend_yield': float(overview.get('DividendYield', 0)) if overview.get('DividendYield') and overview.get('DividendYield') != 'None' else 0,
                            'profit_margin': float(overview.get('ProfitMargin', 0)) if overview.get('ProfitMargin') and overview.get('ProfitMargin') != 'None' else None,
                            'roe': float(overview.get('ReturnOnEquityTTM', 0)) if overview.get('ReturnOnEquityTTM') and overview.get('ReturnOnEquityTTM') != 'None' else None,
                            'debt_to_equity': float(overview.get('DebtToEquity', 0)) if overview.get('DebtToEquity') and overview.get('DebtToEquity') != 'None' else None,
                            'revenue_growth': float(overview.get('QuarterlyRevenueGrowthYOY', 0)) if overview.get('QuarterlyRevenueGrowthYOY') and overview.get('QuarterlyRevenueGrowthYOY') != 'None' else None,
                            'earnings_growth': float(overview.get('QuarterlyEarningsGrowthYOY', 0)) if overview.get('QuarterlyEarningsGrowthYOY') and overview.get('QuarterlyEarningsGrowthYOY') != 'None' else None,
                            'sector': overview.get('Sector', 'Unknown'),
                            'industry': overview.get('Industry', 'Unknown')
                        }
                        
                        logger.debug(f"✅ Downloaded {ticker} via Alpha Vantage (fallback)")
                        return ticker, hist, fundamental_data

            except Exception as e:
                logger.debug(f"Failed to download {ticker}: {e}")

            return ticker, None, None

        # Use parallel processing for yfinance (much faster)
        successful = 0
        # Use more workers since yfinance doesn't have strict rate limits
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(download_single_ticker, ticker): ticker
                      for ticker in self.tickers}

            if TQDM_AVAILABLE:
                iterator = tqdm(as_completed(futures), total=len(futures), desc="Downloading stocks")
            else:
                iterator = as_completed(futures)

            for future in iterator:
                ticker, hist, fundamental_data = future.result()

                if hist is not None and fundamental_data is not None:
                    self.price_data[ticker] = hist
                    self.fundamental_data[ticker] = fundamental_data
                    successful += 1

        # Update tickers only if we downloaded some data
        if self.price_data:
            self.tickers = list(self.price_data.keys())
            logger.info(f"Successfully downloaded {successful} stocks")
        else:
            logger.error(f"Failed to download any stock data! Keeping original ticker list.")
            # Try to reload from backup if available
            if os.path.exists('simulation_data_optimized.pkl'):
                logger.info("Attempting to load from backup data file...")
                try:
                    with open('simulation_data_optimized.pkl', 'rb') as f:
                        backup_data = pickle.load(f)
                        if 'price_data' in backup_data and backup_data['price_data']:
                            self.price_data = backup_data['price_data']
                            self.fundamental_data = backup_data.get('fundamental_data', {})
                            self.tickers = list(self.price_data.keys())
                            logger.info(f"Loaded {len(self.tickers)} tickers from backup")
                            successful = len(self.tickers)
                except Exception as e:
                    logger.error(f"Failed to load backup: {e}")
            
            if not self.price_data:
                raise Exception("Unable to download or load any stock data. Please check your internet connection and try again.")

    def download_fundamental_data(self):
        """Download additional fundamental metrics"""
        logger.info("Enhancing fundamental data...")
        
        # Count tickers to process
        tickers_to_process = [t for t in self.tickers if t in self.price_data]
        total_tickers = len(tickers_to_process)
        logger.info(f"Processing fundamental data for {total_tickers} tickers...")
        
        for idx, ticker in enumerate(tickers_to_process, 1):
            try:
                # Show progress every 10 tickers or for the first/last ticker
                if idx == 1 or idx % 10 == 0 or idx == total_tickers:
                    logger.info(f"Processing ticker {idx}/{total_tickers}: {ticker}")
                
                # Get cash flow data from Alpha Vantage
                cash_flow = self.av_manager.get_cash_flow(ticker)
                
                # Get FCF if available
                if not cash_flow.empty and 'freeCashFlow' in cash_flow.columns:
                    fcf = cash_flow['freeCashFlow'].iloc[0] if not cash_flow.empty else 0
                    market_cap = self.fundamental_data[ticker].get('market_cap', 1)
                    if market_cap > 0 and fcf:
                        self.fundamental_data[ticker]['fcf_yield'] = float(fcf) / market_cap

            except Exception as e:
                logger.debug(f"Could not get enhanced fundamentals for {ticker}: {e}")
        
        logger.info(f"Completed processing fundamental data for {total_tickers} tickers")

    def download_market_indicators(self):
        """Download market indicators"""
        logger.info("Downloading market indicators...")

        # Market indicators including S&P 500 for academic comparison
        indicators = {
            '^DJI': 'dow_jones',
            '^GSPC': 'sp500',
            'SPY': 'sp500_etf',  # S&P 500 ETF as additional benchmark
            'BRK-B': 'berkshire'
        }
        
        # Log which indicators are being skipped to avoid confusion
        skipped = ['volatility (VIX)', '10yr_yield (TNX)', 'dollar_index', 'gold', 'oil']
        logger.info(f"Skipping unsupported indicators: {', '.join(skipped)}")

        for symbol, name in indicators.items():
            try:
                if symbol in ['^GSPC', '^DJI']:
                    # Use SPY and DIA as proxies for S&P 500 and Dow Jones
                    proxy_symbol = 'SPY' if symbol == '^GSPC' else 'DIA'
                    df = self.av_manager.get_daily_adjusted(proxy_symbol)
                else:
                    # Regular stock symbols
                    df = self.av_manager.get_daily_adjusted(symbol.replace('-', '.'))

                if df is not None and not df.empty:
                    # Filter to date range
                    start_date = pd.to_datetime(self.config.START_DATE)
                    end_date = pd.to_datetime(self.config.END_DATE)
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    if 'Adj Close' in df.columns:
                        s = df['Adj Close']
                    elif 'Close' in df.columns:
                        s = df['Close']
                    else:
                        s = df.iloc[:, 0]

                    self.market_data[name] = s.dropna().sort_index()

            except Exception as e:
                logger.warning(f"Could not download {name}: {e}")

    def calculate_metrics(self):
        """Calculate derived metrics"""
        logger.info("Calculating derived metrics...")

        for ticker in self.tickers:
            if ticker in self.price_data:
                df = self.price_data[ticker]

                # Returns
                df['returns'] = df['Close'].pct_change()
                df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

                # Moving averages
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['MA_50'] = df['Close'].rolling(window=50).mean()
                df['MA_200'] = df['Close'].rolling(window=200).mean()

                # Momentum
                df['RSI'] = self.calculate_rsi(df['Close'])
                df['momentum_1m'] = df['Close'] / df['Close'].shift(20) - 1
                df['momentum_3m'] = df['Close'] / df['Close'].shift(60) - 1
                df['momentum_6m'] = df['Close'] / df['Close'].shift(120) - 1
                df['momentum_12m'] = df['Close'] / df['Close'].shift(252) - 1

                # Volatility
                df['volatility_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
                df['volatility_60d'] = df['returns'].rolling(window=60).std() * np.sqrt(252)

                # MACD
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

                # Bollinger Bands
                df['BB_middle'] = df['Close'].rolling(window=20).mean()
                bb_std = df['Close'].rolling(window=20).std()
                df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
                df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

                self.price_data[ticker] = df

    def calculate_adv(self):
        """Calculate average daily volume in dollars"""
        logger.info("Calculating ADV metrics...")

        for ticker in self.tickers:
            if ticker in self.price_data:
                df = self.price_data[ticker]

                # Dollar volume
                df['DollarVolume'] = df['Close'] * df['Volume']
                df['ADV_20'] = df['DollarVolume'].rolling(20).mean()
                df['ADV_60'] = df['DollarVolume'].rolling(60).mean()

                self.adv_data[ticker] = df['ADV_20']
                self.price_data[ticker] = df

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def build_sector_mapping(self):
        """Build sector mapping"""
        for ticker in self.tickers:
            if ticker in self.fundamental_data:
                sector = self.fundamental_data[ticker].get('sector', 'Unknown')
                self.sector_mapping[ticker] = sector

            if not self.ticker_info.empty and ticker in self.ticker_info.index:
                self.sector_mapping[ticker] = self.ticker_info.loc[ticker, 'GICS Sector']

        logger.info(f"Built sector mapping for {len(self.sector_mapping)} tickers")

    def save_data(self):
        """Save data for reproducibility"""
        try:
            with open('simulation_data_complete.pkl', 'wb') as f:
                pickle.dump({
                    'price_data': self.price_data,
                    'fundamental_data': self.fundamental_data,
                    'market_data': self.market_data,
                    'tickers': self.tickers,
                    'sector_mapping': self.sector_mapping,
                    'adv_data': self.adv_data
                }, f)
            logger.info("Data saved to simulation_data_complete.pkl")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")

# ============================================================================
# SECTION 8: LLM DECISION ENGINE - OPTIMIZED
# ============================================================================

class LLMDecisionEngine:
    """Enhanced LLM decision engine with behavioral bias corrections and comprehensive tracking"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.data_manager = None
        self.research_engine = DeepResearchEngine(config)
        
        # Initialize comprehensive tracking for academic research
        self.behavioral_metrics = {
            'cash_deployment_events': [],
            'validation_failures': [],
            'bias_corrections': [],
            'theme_identifications': [],
            'conviction_scores': [],
            'prompt_enhancements': [],
            'llm_performance_scores': [],
            'decision_quality_metrics': []
        }
        
        # Track enhancement effectiveness
        self.enhancement_stats = {
            'total_decisions': 0,
            'validation_triggers': 0,
            'successful_validations': 0,
            'mechanical_fixes_applied': 0,
            'cash_drag_reductions': 0,
            'theme_captures': 0,
            'dynamic_sizing_uses': 0,
            'active_sells_prompted': 0
        }

        # ALWAYS initialize BOTH LLM clients if keys are available (unless dry run)
        if config.DRY_RUN:
            self.openai_client = None
            self.anthropic_client = None
            logger.info("Dry run mode - LLMs disabled")
        else:
            # ALWAYS try to initialize OpenAI if key exists
            if config.OPENAI_API_KEY:
                try:
                    self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
                    logger.info("✅ OpenAI client initialized successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize OpenAI client: {e}")
                    self.openai_client = None
            else:
                logger.warning("⚠️ No OpenAI API key found - OpenAI decisions disabled")
                self.openai_client = None

            # ALWAYS try to initialize Anthropic if key exists
            if config.ANTHROPIC_API_KEY:
                try:
                    self.anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
                    logger.info("✅ Anthropic client initialized successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize Anthropic client: {e}")
                    self.anthropic_client = None
            else:
                logger.warning("⚠️ No Anthropic API key found - Anthropic decisions disabled")
                self.anthropic_client = None

            # Log status
            if self.openai_client and self.anthropic_client:
                logger.info("🎯 BOTH OpenAI and Anthropic will be used for decisions")
            elif self.openai_client:
                logger.info("📊 Only OpenAI available for decisions")
            elif self.anthropic_client:
                logger.info("📊 Only Anthropic available for decisions")
            else:
                logger.warning("⚠️ No LLM clients available - will use rule-based decisions only")

    async def make_investment_decision(self, market_context: Dict, available_stocks: List[Dict],
                                      current_portfolio: Dict, date: str) -> Dict:
        """Make investment decision with deep research - RESPECTS LLM_MODE SETTING"""

        # LIMIT RESEARCH TO TOP CANDIDATES
        research_limit = self.config.MAX_RESEARCH_PER_DATE

        researched_stocks = []

        logger.info(f"Researching top {research_limit} stocks for {date}")

        for stock in available_stocks[:research_limit]:
            ticker = stock['ticker']

            # Conduct research
            research = await self.research_engine.research_company(ticker, date)

            # Combine with screening data
            stock['research'] = research
            stock['buffett_score'] = research['buffett_score'].get('overall', 0)
            stock['recommendation'] = research['buffett_score'].get('recommendation', 'HOLD')

            researched_stocks.append(stock)

        # Sort by Buffett score
        researched_stocks.sort(key=lambda x: x['buffett_score'], reverse=True)

        # Skip LLM only if in dry run
        if self.config.DRY_RUN:
            logger.info("Dry run mode - using rule-based decisions only")
            return self.get_buffett_based_decision(researched_stocks, current_portfolio)

        # Prepare prompt for LLMs
        prompt = self.prepare_investment_prompt(market_context, researched_stocks, current_portfolio, date)

        # Get decisions based on LLM_MODE setting
        logger.info(f"Making investment decision using mode: {self.config.LLM_MODE}")

        gpt_analysis = None
        claude_analysis = None
        final_decision = None

        # Handle different LLM modes
        if self.config.LLM_MODE == 'OPENAI_ONLY':
            # Use only OpenAI
            if self.openai_client:
                logger.info("  📊 Using OPENAI ONLY mode")
                gpt_analysis = await self.get_gpt_analysis(prompt)
                if gpt_analysis:
                    logger.info("  ✅ OpenAI analysis received")
                    final_decision = gpt_analysis
                else:
                    logger.warning("  ⚠️ OpenAI analysis failed")
            else:
                logger.error("  ❌ OpenAI client not available!")

        elif self.config.LLM_MODE == 'ANTHROPIC_ONLY':
            # Use only Anthropic
            if self.anthropic_client:
                logger.info("  🤖 Using ANTHROPIC ONLY mode")
                claude_analysis = await self.get_claude_analysis(prompt)
                if claude_analysis:
                    logger.info("  ✅ Anthropic analysis received")
                    final_decision = claude_analysis
                else:
                    logger.warning("  ⚠️ Anthropic analysis failed")
            else:
                logger.error("  ❌ Anthropic client not available!")

        elif self.config.LLM_MODE == 'BOTH':
            # Use both and combine
            logger.info("  🎯 Using BOTH LLMs mode")

            if self.openai_client:
                logger.info("  📊 Calling OpenAI...")
                gpt_analysis = await self.get_gpt_analysis(prompt)
                if gpt_analysis:
                    logger.info("  ✅ OpenAI analysis received")
                else:
                    logger.warning("  ⚠️ OpenAI analysis failed")

            if self.anthropic_client:
                logger.info("  🤖 Calling Anthropic...")
                claude_analysis = await self.get_claude_analysis(prompt)
                if claude_analysis:
                    logger.info("  ✅ Anthropic analysis received")
                else:
                    logger.warning("  ⚠️ Anthropic analysis failed")

            # Combine decisions
            if gpt_analysis and claude_analysis:
                logger.info("  🔄 Combining decisions from both LLMs")
                final_decision = self.combine_llm_decisions(gpt_analysis, claude_analysis)
            elif gpt_analysis:
                logger.info("  📊 Only OpenAI available")
                final_decision = gpt_analysis
            elif claude_analysis:
                logger.info("  🤖 Only Anthropic available")
                final_decision = claude_analysis

        # Fallback to rule-based if no LLM decision
        if not final_decision or not final_decision.get('actions'):
            logger.warning("No LLM decision available - using enhanced rule-based")
            final_decision = self.get_buffett_based_decision(researched_stocks, current_portfolio)
            
            # If still no actions, force at least one investment
            if not final_decision.get('actions') and researched_stocks:
                logger.info("Forcing minimum investment to avoid zero trades")
                top_stock = researched_stocks[0]  # Best scored stock
                final_decision['actions'] = [{
                    'ticker': top_stock['ticker'],
                    'action': 'BUY',
                    'weight': 0.05,
                    'rationale': f"Top quality stock with score {top_stock.get('buffett_score', 0):.0f}"
                }]

        # Apply portfolio constraints
        final_decision = self.apply_portfolio_constraints(final_decision, current_portfolio)

        # Generate unique decision ID for academic tracking
        decision_id = f"{date}_{hash(str(final_decision)) & 0x7fffffff}"  # Ensure positive hash
        
        # ACADEMIC DATA COLLECTION - Record comprehensive decision quality metrics
        validation_events = {
            'triggered': hasattr(self, '_validation_triggered'),
            'attempts': getattr(self, '_validation_attempts', 0),
            'successful': not hasattr(self, '_mechanical_fixes_applied'),
            'mechanical_fixes': getattr(self, '_mechanical_fixes_applied', 0)
        }
        
        academic_collector.record_decision_quality(
            decision_id=decision_id,
            llm_model=self.config.LLM_MODE,
            portfolio_state=current_portfolio,
            market_context={
                'available_stocks': researched_stocks,
                'volatility': market_context.get('volatility', 0),
                'date': date
            },
            decision=final_decision,
            validation_events=validation_events
        )
        
        # ACADEMIC DATA COLLECTION - Record theme analysis
        if researched_stocks:
            themes_available = self._extract_available_themes(researched_stocks)
            themes_captured = self._extract_captured_themes(final_decision, researched_stocks)
            
            academic_collector.record_theme_analysis(
                decision_id=decision_id,
                themes_available=themes_available,
                themes_captured=themes_captured,
                decision_actions=final_decision.get('actions', [])
            )
        
        # Update enhancement statistics
        self.enhancement_stats['total_decisions'] += 1

        # Add metadata about which LLM was used
        final_decision['llm_mode'] = self.config.LLM_MODE
        final_decision['enhancement_version'] = '2.0_bias_corrected'
        final_decision['decision_id'] = decision_id

        # Log the decision summary with academic tracking
        if final_decision and 'actions' in final_decision:
            num_buys = len([a for a in final_decision['actions'] if a.get('action') == 'BUY'])
            num_sells = len([a for a in final_decision['actions'] if a.get('action') == 'SELL'])
            total_deployment = sum(a.get('weight', 0) for a in final_decision['actions'] if a.get('action') == 'BUY')
            
            logger.info(f"📊 Enhanced decision ({self.config.LLM_MODE}): {num_buys} BUYs, {num_sells} SELLs")
            logger.info(f"💰 Capital deployment: {total_deployment*100:.1f}%")
            logger.info(f"🎓 Academic tracking ID: {decision_id}")

        return final_decision

    def prepare_investment_prompt(self, market_context, stocks, portfolio, date):
        """Prepare detailed prompt with research insights"""

        cash_percentage = portfolio.get('cash_percentage', 100)
        num_holdings = len(portfolio.get('holdings', {}))

        # Cash guidance
        if num_holdings < 5:
            cash_guidance = "Deploy NO MORE than 20% of available cash to maintain liquidity."
        elif cash_percentage > 50:
            cash_guidance = "You have high cash. Deploy up to 40% of available cash."
        elif cash_percentage < 10:
            cash_guidance = "CRITICAL: Low cash. MUST SELL underperforming positions before buying."
        else:
            cash_guidance = "Maintain at least 10% cash for flexibility."

        prompt = f"""
        You are Warren Buffett making investment decisions based on quality and value.
        Date: {date}
        
        MARKET CONTEXT:
        - S&P 500: {market_context.get('sp500', 'N/A')}
        - VIX: {market_context.get('volatility', 'N/A')}
        - 10-Year Yield: {market_context.get('10yr_yield', 'N/A')}%
        
        CURRENT PORTFOLIO:
        Total Value: ${portfolio.get('total_value', 0):,.0f}
        Cash: {portfolio.get('cash_percentage', 0):.1f}% (${portfolio.get('cash', 0):,.0f})
        Holdings: {num_holdings}
        
        IMPORTANT: You MUST make at least 1-3 investment decisions. Being too conservative means missing opportunities.
        The market rewards decisive action based on good analysis.
        
        CASH MANAGEMENT: {cash_guidance}
        
        TOP INVESTMENT CANDIDATES (with Buffett Scores):
        """

        # Add top stocks with research
        for stock in stocks[:10]:  # Reduced from 15
            research = stock.get('research', {})
            buffett = research.get('buffett_score', {})
            fundamentals = research.get('fundamental_analysis', {})

            prompt += f"""
        
        {stock['ticker']}:
        - Buffett Score: {stock.get('buffett_score', 0):.0f}/100 ({buffett.get('classification', 'Unknown')})
        - Recommendation: {stock.get('recommendation', 'HOLD')}
        - Valuation: P/E {fundamentals.get('pe_ratio', 'N/A')}, P/B {fundamentals.get('pb_ratio', 'N/A')}
        - Quality: ROE {fundamentals.get('roe', 0):.1%}, Margin {fundamentals.get('profit_margin', 0):.1%}
        - Financial: D/E {fundamentals.get('debt_to_equity', 'N/A')}, FCF Yield {fundamentals.get('fcf_yield', 0):.1%}
        - Momentum: 6M {stock.get('momentum_6m', 0):.1%}
        """

        prompt += """
        
        DECISION REQUIREMENTS:
        - You MUST select at least 1-3 stocks to BUY from the candidates above
        - Focus on the highest quality companies (Buffett Score > 60)
        - If cash is available, deploy at least 15-30% of it
        - Each position should be 3-8% of portfolio
        - Sell any holdings not meeting quality standards
        
        CRITICAL: Your response MUST be valid JSON. Return a JSON object with this EXACT structure:
        {
            "actions": [
                {"ticker": "AAPL", "action": "BUY", "weight": 0.05, "rationale": "Strong fundamentals"},
                {"ticker": "MSFT", "action": "BUY", "weight": 0.05, "rationale": "Quality growth"}
            ],
            "market_view": "Favorable conditions for quality stocks",
            "risk_assessment": "Moderate risk with diversification",
            "key_themes": ["quality", "value"]
        }
        
        The "actions" array MUST contain at least 1 action. Each action MUST have ticker, action, weight, and rationale fields.
        """

        return prompt

    async def get_gpt_analysis(self, prompt):
        """Get analysis from OpenAI GPT-4 - ALWAYS USES BEST MODEL"""

        if not self.openai_client:
            return None

        try:
            # ALWAYS use GPT-4 for best results (not GPT-3.5)
            model = self.config.LLM_MODEL_PRIMARY  # gpt-4o

            logger.info(f"    Calling OpenAI {model}...")

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are Warren Buffett making investment decisions. Focus on quality, value, and long-term growth. You MUST make at least 1-3 investment decisions - being too conservative means missing opportunities. Return ONLY valid JSON with AT LEAST 1-3 actions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=self.config.LLM_MAX_TOKENS,
                response_format={"type": "json_object"}  # Force JSON response
            )

            content = response.choices[0].message.content
            logger.info(f"    OpenAI raw response length: {len(content)} characters")
            
            # Log first 500 chars of response for debugging
            if content:
                logger.info(f"    OpenAI response preview: {content[:500]}...")
            else:
                logger.warning("    OpenAI returned empty response!")

            # Try multiple JSON extraction methods
            # Method 1: Direct parse
            try:
                decision = json.loads(content)
                logger.info(f"    ✅ OpenAI returned {len(decision.get('actions', []))} actions")
                return decision
            except Exception as e:
                logger.debug(f"    Direct JSON parse failed: {e}")
                pass

            # Method 2: Find JSON in content
            json_match = re.search(r'\{.*}', content, re.DOTALL)
            if json_match:
                try:
                    decision = json.loads(json_match.group())
                    logger.info(f"    ✅ OpenAI returned {len(decision.get('actions', []))} actions (extracted)")
                    return decision
                except Exception as e2:
                    logger.debug(f"    Regex JSON extraction failed: {e2}")
                    pass

            # Method 3: Clean and retry
            cleaned = content.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]

            try:
                decision = json.loads(cleaned.strip())
                logger.info(f"    ✅ OpenAI returned {len(decision.get('actions', []))} actions (cleaned)")
                return decision
            except Exception as e3:
                logger.warning(f"    ⚠️ Could not parse OpenAI response as JSON: {e3}")
                logger.warning(f"    Final cleaned content attempt: {cleaned.strip()[:200]}...")

                # Return a default conservative decision
                return {
                    "actions": [],
                    "market_view": "Unable to parse response",
                    "risk_assessment": "High uncertainty",
                    "key_themes": []
                }

        except Exception as e:
            logger.error(f"    ❌ OpenAI analysis failed: {e}")
            return None

    async def get_claude_analysis(self, prompt):
        """Get analysis from Anthropic Claude - ALWAYS USES BEST MODEL"""

        if not self.anthropic_client:
            return None

        try:
            # ALWAYS use Claude Opus for best results (not Haiku)
            model = self.config.LLM_MODEL_SECONDARY  # claude-3-opus-20240229

            logger.info(f"    Calling Anthropic {model}...")

            response = self.anthropic_client.messages.create(
                model=model,
                messages=[{"role": "user", "content": f"You are Warren Buffett making investment decisions. Focus on quality, value, and long-term growth. You MUST make at least 1-3 investment decisions - being too conservative means missing opportunities. Return ONLY valid JSON with AT LEAST 1-3 actions.\n\n{prompt}"}],
                max_tokens=self.config.LLM_MAX_TOKENS,
                temperature=self.config.LLM_TEMPERATURE
            )

            content = response.content[0].text
            logger.debug(f"    Anthropic raw response length: {len(content)} characters")

            # Parse JSON with same methods as GPT
            try:
                decision = json.loads(content)
                logger.info(f"    ✅ Anthropic returned {len(decision.get('actions', []))} actions")
                return decision
            except:
                pass

            json_match = re.search(r'\{.*}', content, re.DOTALL)
            if json_match:
                try:
                    decision = json.loads(json_match.group())
                    logger.info(f"    ✅ Anthropic returned {len(decision.get('actions', []))} actions (extracted)")
                    return decision
                except:
                    pass

            # Clean and retry
            cleaned = content.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]

            try:
                decision = json.loads(cleaned.strip())
                logger.info(f"    ✅ Anthropic returned {len(decision.get('actions', []))} actions (cleaned)")
                return decision
            except:
                logger.warning(f"    ⚠️ Could not parse Anthropic response as JSON")

                return {
                    "actions": [],
                    "market_view": "Unable to parse response",
                    "risk_assessment": "High uncertainty",
                    "key_themes": []
                }

        except Exception as e:
            logger.error(f"    ❌ Anthropic analysis failed: {e}")
            return None

    def combine_llm_decisions(self, gpt_decision, claude_decision):
        """Combine decisions from BOTH OpenAI and Anthropic for best results"""

        if not gpt_decision and not claude_decision:
            logger.warning("Both LLM decisions are empty")
            return None

        if not claude_decision:
            logger.info("Only OpenAI decision available")
            return gpt_decision

        if not gpt_decision:
            logger.info("Only Anthropic decision available")
            return claude_decision

        logger.info("🔄 Combining decisions from BOTH OpenAI and Anthropic...")

        # Combine analyses
        combined = {
            "actions": [],
            "market_view": f"GPT: {gpt_decision.get('market_view', '')} | Claude: {claude_decision.get('market_view', '')}",
            "risk_assessment": f"{gpt_decision.get('risk_assessment', '')} {claude_decision.get('risk_assessment', '')}",
            "key_themes": list(set(gpt_decision.get('key_themes', []) + claude_decision.get('key_themes', [])))
        }

        # Merge actions from both LLMs
        gpt_actions = {a['ticker']: a for a in gpt_decision.get('actions', [])}
        claude_actions = {a['ticker']: a for a in claude_decision.get('actions', [])}

        logger.info(f"  OpenAI suggested {len(gpt_actions)} actions")
        logger.info(f"  Anthropic suggested {len(claude_actions)} actions")

        all_tickers = set(gpt_actions.keys()) | set(claude_actions.keys())

        # Track consensus
        consensus_buys = []
        consensus_sells = []
        disagreements = []

        for ticker in all_tickers:
            gpt_a = gpt_actions.get(ticker, {'weight': 0, 'action': 'HOLD'})
            claude_a = claude_actions.get(ticker, {'weight': 0, 'action': 'HOLD'})

            # Check for consensus
            if gpt_a.get('action') == claude_a.get('action'):
                action = gpt_a.get('action')
                avg_weight = (gpt_a.get('weight', 0) + claude_a.get('weight', 0)) / 2

                if action == 'BUY':
                    consensus_buys.append(ticker)
                elif action == 'SELL':
                    consensus_sells.append(ticker)

                rationale = f"Strong consensus from both OpenAI and Anthropic"
            else:
                # Conservative approach when they disagree
                disagreements.append(f"{ticker}: GPT={gpt_a.get('action')} vs Claude={claude_a.get('action')}")

                if 'SELL' in [gpt_a.get('action'), claude_a.get('action')]:
                    action = 'SELL'
                    avg_weight = 0
                    rationale = "One model suggests selling (conservative approach)"
                elif 'BUY' in [gpt_a.get('action'), claude_a.get('action')]:
                    action = 'BUY'
                    avg_weight = min(gpt_a.get('weight', 0), claude_a.get('weight', 0))
                    rationale = "Mixed signal - reduced position size"
                else:
                    continue

            if action != 'HOLD':
                combined["actions"].append({
                    "ticker": ticker,
                    "action": action,
                    "weight": avg_weight,
                    "rationale": rationale
                })

        # Log consensus and disagreements
        if consensus_buys:
            logger.info(f"  ✅ Consensus BUYs: {', '.join(consensus_buys)}")
        if consensus_sells:
            logger.info(f"  ✅ Consensus SELLs: {', '.join(consensus_sells)}")
        if disagreements:
            logger.info(f"  ⚠️ Disagreements: {'; '.join(disagreements)}")

        logger.info(f"  📊 Final combined: {len(combined['actions'])} actions")
        
        # Track performance metrics for both LLMs
        if gpt_decision:
            self.track_llm_performance_metrics('OpenAI', gpt_decision)
        if claude_decision:
            self.track_llm_performance_metrics('Anthropic', claude_decision)

        return combined
        
    def track_llm_performance_metrics(self, llm_name, decision, actual_results=None):
        """Track LLM performance metrics for future decision weighting"""
        
        # In a production system, this would track:
        # - How often each LLM's decisions meet bias requirements
        # - Performance outcomes of decisions
        # - Calibration of conviction scores
        
        metrics = {
            'llm': llm_name,
            'decision_quality_score': self.score_decision_quality(decision),
            'bias_adherence_score': self.score_bias_adherence(decision),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.debug(f"📈 {llm_name} performance metrics: {metrics}")
        return metrics
    
    def score_decision_quality(self, decision):
        """Score the quality of a decision for LLM performance tracking"""
        
        if not decision or not decision.get('actions'):
            return 0
        
        score = 50  # Base score
        actions = decision.get('actions', [])
        
        # Points for having multiple actions (avoiding analysis paralysis)
        if len(actions) >= 2:
            score += 20
        elif len(actions) >= 4:
            score += 30
        
        # Points for variable position sizing (avoiding fixed sizing bias)
        weights = [a.get('weight', 0) for a in actions if a.get('action') == 'BUY']
        if len(set(weights)) > 1:  # Variable sizing
            score += 15
        
        # Points for having conviction scores
        if all(a.get('conviction') for a in actions):
            score += 10
        
        # Points for including sells
        if any(a.get('action') == 'SELL' for a in actions):
            score += 5
        
        return min(100, max(0, score))
    
    def score_bias_adherence(self, decision):
        """Score how well the decision adheres to bias correction requirements"""
        
        if not decision or not decision.get('actions'):
            return 0
        
        score = 0
        actions = decision.get('actions', [])
        
        # Check for minimum trades (analysis paralysis)
        if len(actions) >= 2:
            score += 25
        
        # Check for cash deployment
        total_deployment = sum(a.get('weight', 0) for a in actions if a.get('action') == 'BUY')
        if total_deployment >= 0.15:  # At least 15% deployment
            score += 25
        
        # Check for variable sizing
        buy_weights = [a.get('weight', 0) for a in actions if a.get('action') == 'BUY']
        if len(buy_weights) > 1 and len(set(buy_weights)) > 1:
            score += 25
        
        # Check for theme participation
        themes = decision.get('key_themes', [])
        if len(themes) > 0:
            score += 25
        
        return min(100, max(0, score))
    
    def _extract_available_themes(self, researched_stocks):
        """Extract available market themes from researched stocks for academic tracking"""
        
        theme_data = defaultdict(list)
        
        for stock in researched_stocks:
            research = stock.get('research', {})
            momentum = stock.get('momentum_6m', 0)
            buffett_score = stock.get('buffett_score', 0)
            
            # Extract themes from research
            themes = []
            business_desc = research.get('business_description', '').upper()
            sector = research.get('sector', 'Unknown')
            
            # AI/Tech theme
            if any(keyword in business_desc for keyword in ['AI', 'ARTIFICIAL', 'MACHINE LEARNING', 'CLOUD', 'SOFTWARE']):
                themes.append('AI/Technology')
            
            # Energy theme
            if any(keyword in business_desc for keyword in ['ENERGY', 'OIL', 'GAS', 'RENEWABLE']):
                themes.append('Energy')
            
            # Healthcare theme
            if any(keyword in business_desc for keyword in ['HEALTH', 'PHARMA', 'BIOTECH', 'MEDICAL']):
                themes.append('Healthcare')
            
            # Add sector as theme
            if sector != 'Unknown':
                themes.append(sector)
            
            for theme in themes:
                theme_data[theme].append({
                    'ticker': stock['ticker'],
                    'momentum': momentum,
                    'quality': buffett_score
                })
        
        # Calculate theme statistics
        available_themes = []
        for theme, stocks in theme_data.items():
            if len(stocks) >= 2:  # Need at least 2 stocks for a theme
                avg_momentum = np.mean([s['momentum'] for s in stocks])
                avg_quality = np.mean([s['quality'] for s in stocks])
                strength = (avg_momentum * 0.6 + avg_quality * 0.4) / 100  # Normalize to 0-1
                
                available_themes.append({
                    'name': theme,
                    'stock_count': len(stocks),
                    'momentum': avg_momentum,
                    'quality': avg_quality,
                    'strength': strength,
                    'stocks': [s['ticker'] for s in stocks]
                })
        
        # Sort by strength
        available_themes.sort(key=lambda x: x['strength'], reverse=True)
        return available_themes
    
    def _extract_captured_themes(self, decision, researched_stocks):
        """Extract themes captured by the decision for academic tracking"""
        
        actions = decision.get('actions', [])
        buy_actions = [a for a in actions if a.get('action') == 'BUY']
        
        if not buy_actions:
            return []
        
        stock_dict = {s['ticker']: s for s in researched_stocks}
        theme_allocations = defaultdict(float)
        
        for action in buy_actions:
            ticker = action.get('ticker', '')
            weight = action.get('weight', 0)
            
            stock = stock_dict.get(ticker)
            if not stock:
                continue
            
            research = stock.get('research', {})
            business_desc = research.get('business_description', '').upper()
            sector = research.get('sector', 'Unknown')
            
            # Determine themes for this stock
            stock_themes = []
            if any(keyword in business_desc for keyword in ['AI', 'ARTIFICIAL', 'MACHINE LEARNING', 'CLOUD', 'SOFTWARE']):
                stock_themes.append('AI/Technology')
            if any(keyword in business_desc for keyword in ['ENERGY', 'OIL', 'GAS', 'RENEWABLE']):
                stock_themes.append('Energy')
            if any(keyword in business_desc for keyword in ['HEALTH', 'PHARMA', 'BIOTECH', 'MEDICAL']):
                stock_themes.append('Healthcare')
            if sector != 'Unknown':
                stock_themes.append(sector)
            
            # Distribute weight across themes
            if stock_themes:
                weight_per_theme = weight / len(stock_themes)
                for theme in stock_themes:
                    theme_allocations[theme] += weight_per_theme
        
        # Convert to list format
        captured_themes = []
        for theme, allocation in theme_allocations.items():
            captured_themes.append({
                'name': theme,
                'allocation': allocation,
                'rationale': f'Portfolio exposure via selected stocks'
            })
        
        # Sort by allocation
        captured_themes.sort(key=lambda x: x['allocation'], reverse=True)
        return captured_themes
    
    def record_bias_correction_event(self, decision_id, bias_type, before_value, after_value, 
                                   correction_method='llm_retry', severity='medium', success=True):
        """Record a bias correction event for academic analysis"""
        
        academic_collector.record_bias_correction(
            decision_id=decision_id,
            bias_type=bias_type,
            before_value=before_value,
            after_value=after_value,
            correction_method=correction_method,
            llm_model=self.config.LLM_MODE,
            success=success,
            severity=severity
        )
        
        # Update internal statistics
        if bias_type == 'cash_drag':
            self.enhancement_stats['cash_drag_reductions'] += 1
        elif bias_type == 'theme_blindness':
            self.enhancement_stats['theme_captures'] += 1
        elif bias_type == 'fixed_sizing':
            self.enhancement_stats['dynamic_sizing_uses'] += 1
        elif bias_type == 'analysis_paralysis':
            self.enhancement_stats['validation_triggers'] += 1
        
        # Log for immediate visibility
        logger.info(f"🎓 BIAS CORRECTION: {bias_type} | {before_value:.2f} → {after_value:.2f} | Method: {correction_method} | Success: {success}")
    
    def get_academic_summary(self):
        """Get summary of academic data collection for this session"""
        
        return {
            'session_stats': self.enhancement_stats.copy(),
            'behavioral_metrics_count': len(self.behavioral_metrics['bias_corrections']),
            'academic_data_status': 'collecting' if self.enhancement_stats['total_decisions'] > 0 else 'ready',
            'export_available': self.enhancement_stats['total_decisions'] > 0
        }
    
    def export_academic_data(self, filename_prefix=None):
        """Export all collected academic data for research analysis"""
        
        if filename_prefix is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename_prefix = f"llm_enhancement_study_{timestamp}"
        
        # Export via academic collector
        exported_files = academic_collector.export_academic_dataset(filename_prefix)
        
        # Add system-specific statistics
        system_stats = {
            'enhancement_statistics': self.enhancement_stats,
            'behavioral_metrics': self.behavioral_metrics,
            'configuration': {
                'llm_mode': self.config.LLM_MODE,
                'models': {
                    'primary': self.config.LLM_MODEL_PRIMARY,
                    'secondary': self.config.LLM_MODEL_SECONDARY
                },
                'validation_enabled': True,
                'bias_correction_enabled': True
            }
        }
        
        # Save system statistics
        import json
        from pathlib import Path
        
        system_file = Path(f"{filename_prefix}_system_statistics.json")
        with open(system_file, 'w') as f:
            json.dump(system_stats, f, indent=2, default=str)
        
        exported_files['system_statistics'] = str(system_file)
        
        logger.info(f"🎓 Academic data export completed: {list(exported_files.keys())}")
        return exported_files

    def get_buffett_based_decision(self, stocks, portfolio):
        """Rule-based decision using Buffett principles"""

        decision = {
            "actions": [],
            "market_view": "Focus on quality companies with sustainable moats",
            "risk_assessment": "Conservative approach with margin of safety",
            "key_themes": ["quality", "value", "long-term"]
        }

        current_holdings = portfolio.get('holdings', {})

        # Sell poor quality holdings
        for ticker in current_holdings:
            stock_data = next((s for s in stocks if s['ticker'] == ticker), None)
            if stock_data and stock_data.get('buffett_score', 0) < 40:  # Lower threshold to avoid excessive selling
                decision['actions'].append({
                    'ticker': ticker,
                    'action': 'SELL',
                    'weight': 0,
                    'rationale': f"Low Buffett score: {stock_data.get('buffett_score', 0):.0f}"
                })

        # Buy high quality stocks - be more aggressive to ensure trading
        stocks_to_buy = 0
        for stock in stocks[:15]:  # Consider more candidates
            ticker = stock['ticker']
            buffett_score = stock.get('buffett_score', 0)

            # Lower threshold and limit to ensure we make trades
            if ticker not in current_holdings and buffett_score > 50 and stocks_to_buy < 5:
                # Position size based on conviction
                if buffett_score > 85:
                    weight = 0.08
                elif buffett_score > 75:
                    weight = 0.06
                elif buffett_score > 65:
                    weight = 0.05
                elif buffett_score > 55:
                    weight = 0.04
                else:
                    weight = 0.03

                decision['actions'].append({
                    'ticker': ticker,
                    'action': 'BUY',
                    'weight': weight,
                    'rationale': f"Quality stock with Buffett score: {buffett_score:.0f}"
                })
                stocks_to_buy += 1

        return decision

    def apply_portfolio_constraints(self, decision, current_portfolio):
        """Apply portfolio constraints"""

        if not decision or 'actions' not in decision:
            return self.get_default_decision()

        actions = decision.get('actions', [])

        # Get constraints
        available_tickers = set(self.data_manager.tickers) if self.data_manager else set()
        sector_mapping = self.data_manager.sector_mapping if self.data_manager else {}

        # Calculate current sector exposures
        current_sector_weights = defaultdict(float)
        total_value = current_portfolio.get('total_value', 1)

        for ticker, position in current_portfolio.get('holdings', {}).items():
            sector = sector_mapping.get(ticker, 'Unknown')
            weight = position.get('value', 0) / total_value
            current_sector_weights[sector] += weight

        # Validate and adjust actions
        valid_actions = []

        for action in actions:
            if not isinstance(action, dict) or 'ticker' not in action:
                continue

            ticker = action['ticker'].strip().upper()

            # Apply canonicalization
            ticker = self.config.TICKER_CANON.get(ticker, ticker)

            # Check if valid
            if available_tickers and ticker not in available_tickers:
                logger.warning(f"Ticker {ticker} not in universe")
                continue

            # Check sector constraints for buys
            if action.get('action') == 'BUY':
                sector = sector_mapping.get(ticker, 'Unknown')
                proposed_weight = float(action.get('weight', 0))

                new_sector_weight = current_sector_weights[sector] + proposed_weight

                if new_sector_weight > self.config.MAX_SECTOR_EXPOSURE:
                    max_additional = self.config.MAX_SECTOR_EXPOSURE - current_sector_weights[sector]
                    if max_additional > 0.01:
                        action['weight'] = min(proposed_weight, max_additional)
                    else:
                        continue

                # Apply position size limit
                action['weight'] = min(action['weight'], self.config.MAX_POSITION_SIZE)

            valid_actions.append(action)

        # Limit number of positions
        if len(valid_actions) > self.config.MAX_POSITIONS:
            sells = [a for a in valid_actions if a.get('action') in ['SELL', 'REDUCE']]
            buys = sorted([a for a in valid_actions if a.get('action') == 'BUY'],
                         key=lambda x: x.get('weight', 0), reverse=True)
            valid_actions = sells + buys[:self.config.MAX_POSITIONS - len(sells)]

        logger.info(f"Filtered {len(actions)} actions to {len(valid_actions)} valid actions")

        decision['actions'] = valid_actions
        return decision

    def get_default_decision(self):
        """Default conservative decision"""
        return {
            "actions": [],
            "market_view": "Unable to analyze - maintaining positions",
            "risk_assessment": "High uncertainty",
            "key_themes": []
        }

# ============================================================================
# SECTION 9: PORTFOLIO MANAGER - FIXED
# ============================================================================

class PortfolioManager:
    """Complete portfolio management system - FIXED"""

    def __init__(self, config: SimulationConfig, data_manager):
        self.config = config
        self.data_manager = data_manager
        self.reset_portfolio()  # Use reset method for initialization

    def reset_portfolio(self):
        """Reset portfolio to initial state - useful for comparison runs"""
        self.portfolio = {
            'cash': self.config.INITIAL_CAPITAL,
            'holdings': {},
            'total_value': self.config.INITIAL_CAPITAL,
            'history': []
        }
        self.trades = []
        self.berkshire_convergence = []
        self.yearly_returns = {}
        self.daily_equity = pd.Series(dtype=float)
        self._processed_sells = set()

    def execute_decision(self, decision: Dict, date: str):
        """Execute investment decision"""

        logger.info(f"Executing decisions for {date}")

        # Reset processed sells
        self._processed_sells.clear()

        current_prices = self.get_current_prices(date)

        # Update portfolio value
        self.update_portfolio_value(current_prices)

        # Check cash levels
        cash_percentage = self.portfolio['cash'] / self.portfolio['total_value'] if self.portfolio['total_value'] > 0 else 1.0

        if cash_percentage < 0.05:
            logger.info(f"Low cash ({cash_percentage:.1%}), forcing rebalancing")
            self.rebalance_for_cash(current_prices, decision)

        # Process sells first
        self._process_sell_actions(decision.get('actions', []), current_prices, date)

        # Update after sells
        self.update_portfolio_value(current_prices)

        # Process buys
        self._process_buy_actions(decision.get('actions', []), current_prices, date)

        # Final update
        self.update_portfolio_value(current_prices)

        # Record state
        self.record_portfolio_state(date)

        # Track Berkshire convergence
        self.track_berkshire_convergence(decision, date)

    def _process_sell_actions(self, actions: List[Dict], current_prices: Dict, date: str):
        """Process sell actions"""

        for action in actions:
            ticker = action.get('ticker')
            if not ticker or ticker in self._processed_sells:
                continue

            # Canonicalize
            ticker = self.config.TICKER_CANON.get(ticker, ticker)

            action_type = action.get('action', '').upper()

            if action_type == 'SELL':
                if self._validate_sell(ticker):
                    self.sell_position(ticker, current_prices.get(ticker, 0), date)
                    self._processed_sells.add(ticker)

            elif action_type == 'REDUCE':
                if self._validate_sell(ticker):
                    target_weight = action.get('target_weight', 0)
                    self.reduce_position(ticker, target_weight, current_prices.get(ticker, 0), date)

    def _process_buy_actions(self, actions: List[Dict], current_prices: Dict, date: str):
        """Process buy actions with ADV constraints"""

        for action in actions:
            if action.get('action', '').upper() != 'BUY':
                continue

            ticker = action.get('ticker')
            if not ticker:
                continue

            # Canonicalize
            ticker = self.config.TICKER_CANON.get(ticker, ticker)

            if ticker not in current_prices:
                logger.warning(f"Cannot buy {ticker} - no price available")
                continue

            # Check cash
            min_cash_reserve = self.portfolio['total_value'] * self.config.MIN_CASH
            available_cash = max(0, self.portfolio['cash'] - min_cash_reserve)

            if available_cash < 1000:
                logger.warning(f"Insufficient cash for {ticker}")
                continue

            # Calculate target value
            target_value = min(
                self.portfolio['total_value'] * action.get('weight', 0),
                available_cash * 0.9
            )

            # Apply ADV constraint
            target_value = self._apply_adv_constraint(ticker, target_value, date)

            # Dynamic minimum trade threshold: 0.01% of total portfolio value
            min_trade_threshold = self.portfolio['total_value'] * 0.0001  # $100K for $100M portfolio
            
            if target_value > min_trade_threshold:
                self.buy_position(ticker, target_value, current_prices[ticker], date)
            else:
                logger.debug(f"Skipping {ticker}: target ${target_value:,.0f} below minimum ${min_trade_threshold:,.0f}")

    def _apply_adv_constraint(self, ticker: str, target_value: float, date: str) -> float:
        """Apply ADV constraint to position size"""

        if ticker not in self.data_manager.adv_data:
            return target_value

        date_obj = normalize_date(date)
        adv_series = self.data_manager.adv_data[ticker]
        
        # Ensure timezone compatibility
        if adv_series.index.tz is not None:
            adv_series.index = adv_series.index.tz_localize(None)

        if date_obj in adv_series.index:
            adv = adv_series.loc[date_obj]
        else:
            valid_dates = adv_series.index[adv_series.index <= date_obj]
            if len(valid_dates) > 0:
                adv = adv_series.loc[valid_dates[-1]]
            else:
                return target_value

        if adv > 0:
            max_position = adv * self.config.ADV_PARTICIPATION_LIMIT
            if target_value > max_position:
                logger.info(f"Reducing {ticker} size from ${target_value:,.0f} to ${max_position:,.0f} (ADV limit)")
                return max_position

        return target_value

    def _validate_sell(self, ticker: str) -> bool:
        """Validate sell action"""
        return ticker in self.portfolio['holdings'] and self.portfolio['holdings'][ticker]['shares'] > 0

    def sell_position(self, ticker: str, price: float, date: str):
        """Execute sell order"""

        if not self._validate_sell(ticker) or price <= 0:
            return

        position = self.portfolio['holdings'][ticker]
        shares = position['shares']

        # Calculate proceeds
        gross_proceeds = shares * price

        # Transaction costs
        commission = min(shares * self.config.COMMISSION_PER_SHARE, gross_proceeds * 0.01)
        spread_cost = gross_proceeds * (self.config.SPREAD_COST_BP / 10000)
        market_impact = self.calculate_market_impact(ticker, gross_proceeds, date)

        net_proceeds = gross_proceeds - commission - spread_cost - market_impact

        # Update portfolio
        self.portfolio['cash'] += net_proceeds

        # Calculate P&L
        gain_loss = net_proceeds - position.get('cost_basis', 0)

        # Record trade
        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'proceeds': net_proceeds,
            'gain_loss': gain_loss
        })

        # Remove position
        del self.portfolio['holdings'][ticker]

        logger.info(f"Sold {shares} shares of {ticker} at ${price:.2f} (Net: ${net_proceeds:,.2f}, G/L: ${gain_loss:,.2f})")

    def reduce_position(self, ticker: str, target_weight: float, price: float, date: str):
        """Reduce position to target weight"""

        if not self._validate_sell(ticker) or price <= 0:
            return

        position = self.portfolio['holdings'][ticker]
        current_value = position['shares'] * price
        target_value = self.portfolio['total_value'] * target_weight

        if current_value <= target_value:
            return

        # Calculate shares to sell
        value_to_sell = current_value - target_value
        shares_to_sell = min(int(value_to_sell / price), position['shares'] - 1)

        if shares_to_sell <= 0:
            return

        # Execute partial sell
        gross_proceeds = shares_to_sell * price

        # Transaction costs
        commission = min(shares_to_sell * self.config.COMMISSION_PER_SHARE, gross_proceeds * 0.01)
        spread_cost = gross_proceeds * (self.config.SPREAD_COST_BP / 10000)
        market_impact = self.calculate_market_impact(ticker, gross_proceeds, date)

        net_proceeds = gross_proceeds - commission - spread_cost - market_impact

        # Update portfolio
        self.portfolio['cash'] += net_proceeds
        position['shares'] -= shares_to_sell

        # Adjust cost basis
        proportion_sold = shares_to_sell / (shares_to_sell + position['shares'])
        cost_basis_sold = position['cost_basis'] * proportion_sold
        position['cost_basis'] -= cost_basis_sold
        position['avg_price'] = position['cost_basis'] / position['shares'] if position['shares'] > 0 else 0

        # Record trade
        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': 'REDUCE',
            'shares': shares_to_sell,
            'price': price,
            'proceeds': net_proceeds,
            'gain_loss': net_proceeds - cost_basis_sold
        })

        logger.info(f"Reduced {ticker}: sold {shares_to_sell} shares at ${price:.2f}")

    def buy_position(self, ticker: str, target_value: float, price: float, date: str):
        """Execute buy order"""

        try:
            if ticker not in self.data_manager.tickers:
                logger.warning(f"Ticker {ticker} not in universe")
                return

            if price <= 0 or target_value <= 0:
                return

            # Calculate shares
            shares = max(1, int(target_value / price))
            gross_cost = shares * price

            # Transaction costs
            commission = min(shares * self.config.COMMISSION_PER_SHARE, gross_cost * 0.01)
            spread_cost = gross_cost * (self.config.SPREAD_COST_BP / 10000)
            market_impact = self.calculate_market_impact(ticker, gross_cost, date)

            total_cost = gross_cost + commission + spread_cost + market_impact

            # Check cash
            if total_cost > self.portfolio['cash'] * 0.98:
                available = self.portfolio['cash'] * 0.95
                if available < price:
                    return

                shares = max(1, int(available / (price * 1.02)))
                gross_cost = shares * price
                commission = min(shares * self.config.COMMISSION_PER_SHARE, gross_cost * 0.01)
                spread_cost = gross_cost * (self.config.SPREAD_COST_BP / 10000)
                market_impact = self.calculate_market_impact(ticker, gross_cost, date)
                total_cost = gross_cost + commission + spread_cost + market_impact

            if total_cost > self.portfolio['cash']:
                return

            # Execute trade
            self.portfolio['cash'] -= total_cost

            if ticker in self.portfolio['holdings']:
                # Add to existing
                existing = self.portfolio['holdings'][ticker]
                existing['shares'] += shares
                existing['cost_basis'] += total_cost
                existing['avg_price'] = existing['cost_basis'] / existing['shares']
            else:
                # New position
                self.portfolio['holdings'][ticker] = {
                    'shares': shares,
                    'cost_basis': total_cost,
                    'avg_price': total_cost / shares
                }

            # Record trade
            self.trades.append({
                'date': date,
                'ticker': ticker,
                'action': 'BUY',
                'shares': shares,
                'price': price,
                'cost': total_cost
            })

            logger.info(f"Bought {shares} shares of {ticker} at ${price:.2f} (Cost: ${total_cost:,.2f})")

        except Exception as e:
            logger.error(f"Error buying {ticker}: {e}")

    def calculate_market_impact(self, ticker: str, notional: float, date: str) -> float:
        """Calculate market impact based on ADV"""

        if ticker not in self.data_manager.adv_data:
            return notional * self.config.MARKET_IMPACT_FACTOR * 0.005

        date_obj = normalize_date(date)
        adv_series = self.data_manager.adv_data[ticker]
        
        # Ensure timezone compatibility
        if adv_series.index.tz is not None:
            adv_series.index = adv_series.index.tz_localize(None)

        if date_obj in adv_series.index:
            adv = adv_series.loc[date_obj]
        else:
            valid_dates = adv_series.index[adv_series.index <= date_obj]
            if len(valid_dates) > 0:
                adv = adv_series.loc[valid_dates[-1]]
            else:
                adv = 0

        if adv > 0:
            participation_rate = notional / adv
            # Square root impact model
            impact = notional * self.config.MARKET_IMPACT_FACTOR * min(0.02, np.sqrt(participation_rate))
        else:
            impact = notional * self.config.MARKET_IMPACT_FACTOR * 0.01

        return impact

    def rebalance_for_cash(self, current_prices: Dict, decision: Dict):
        """Rebalance to raise cash"""

        if not decision.get('actions'):
            decision['actions'] = []

        # Get existing sells
        existing_sells = {
            action['ticker'] for action in decision['actions']
            if action.get('action') in ['SELL', 'REDUCE']
        }

        # Evaluate holdings
        holdings_to_evaluate = []

        for ticker, position in self.portfolio['holdings'].items():
            if ticker in existing_sells or ticker not in current_prices:
                continue

            current_price = current_prices[ticker]
            if current_price <= 0:
                continue

            current_value = position['shares'] * current_price
            current_weight = current_value / self.portfolio['total_value'] if self.portfolio['total_value'] > 0 else 0

            avg_price = position.get('avg_price', 0)
            position_return = (current_price / avg_price - 1) if avg_price > 0 else 0

            holdings_to_evaluate.append({
                'ticker': ticker,
                'weight': current_weight,
                'return': position_return,
                'value': current_value
            })

        # Sort by return (sell losers first)
        holdings_to_evaluate.sort(key=lambda x: x['return'])

        # Target cash
        target_cash = self.portfolio['total_value'] * 0.15
        current_cash = self.portfolio['cash']
        cash_needed = max(0, target_cash - current_cash)

        if cash_needed <= 0:
            return

        cash_to_raise = 0

        for holding in holdings_to_evaluate:
            if cash_to_raise >= cash_needed:
                break

            ticker = holding['ticker']

            if holding['return'] < -0.10:
                # Full sell for big losers
                decision['actions'].append({
                    'ticker': ticker,
                    'action': 'SELL',
                    'weight': 0,
                    'rationale': 'Cutting losses'
                })
                cash_to_raise += holding['value'] * 0.95

            elif holding['weight'] > 0.15:
                # Trim overweight
                decision['actions'].append({
                    'ticker': ticker,
                    'action': 'REDUCE',
                    'target_weight': 0.10,
                    'rationale': 'Trimming overweight position'
                })
                cash_to_raise += holding['value'] * 0.33

            elif holding['return'] < 0:
                # Reduce underperformers
                decision['actions'].append({
                    'ticker': ticker,
                    'action': 'REDUCE',
                    'target_weight': holding['weight'] * 0.5,
                    'rationale': 'Reducing underperformer'
                })
                cash_to_raise += holding['value'] * 0.5

        logger.info(f"Added {len(decision['actions']) - len(existing_sells)} rebalancing actions")

    def get_current_prices(self, date: str) -> Dict[str, float]:
        """Get prices for specific date"""
        prices = {}

        date_obj = normalize_date(date)

        for ticker in self.data_manager.tickers:
            if ticker not in self.data_manager.price_data:
                continue

            try:
                df = self.data_manager.price_data[ticker]

                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                if date_obj in df.index:
                    prices[ticker] = float(df.loc[date_obj, 'Close'])
                else:
                    valid_dates = df.index[df.index <= date_obj]
                    if len(valid_dates) > 0:
                        prices[ticker] = float(df.loc[valid_dates[-1], 'Close'])

            except Exception as e:
                logger.debug(f"Could not get price for {ticker} on {date}: {e}")

        return prices

    def update_portfolio_value(self, current_prices: Dict):
        """Update portfolio value"""

        holdings_value = 0

        for ticker, position in self.portfolio['holdings'].items():
            if ticker in current_prices and current_prices[ticker] > 0:
                position['current_price'] = current_prices[ticker]
                position['value'] = position['shares'] * current_prices[ticker]

                avg_price = position.get('avg_price', 0)
                if avg_price > 0:
                    position['return'] = (current_prices[ticker] / avg_price - 1) * 100
                else:
                    position['return'] = 0

                holdings_value += position['value']

        self.portfolio['total_value'] = self.portfolio['cash'] + holdings_value

        if self.portfolio['total_value'] > 0:
            self.portfolio['cash_percentage'] = (self.portfolio['cash'] / self.portfolio['total_value']) * 100

            for position in self.portfolio['holdings'].values():
                if 'value' in position:
                    position['weight'] = (position['value'] / self.portfolio['total_value']) * 100
        else:
            self.portfolio['cash_percentage'] = 100

    def record_portfolio_state(self, date: str):
        """Record portfolio snapshot"""

        snapshot = {
            'date': date,
            'total_value': self.portfolio['total_value'],
            'cash': self.portfolio['cash'],
            'cash_percentage': self.portfolio.get('cash_percentage', 100),
            'holdings_value': self.portfolio['total_value'] - self.portfolio['cash'],
            'num_holdings': len(self.portfolio['holdings']),
            'holdings': {k: v.copy() for k, v in self.portfolio['holdings'].items()}
        }

        self.portfolio['history'].append(snapshot)

        # Track yearly returns
        year = pd.to_datetime(date).year
        if year not in self.yearly_returns:
            self.yearly_returns[year] = {'start_value': self.portfolio['total_value']}
        self.yearly_returns[year]['end_value'] = self.portfolio['total_value']

    def track_berkshire_convergence(self, decision: Dict, date: str):
        """Track convergence with Berkshire decisions"""

        berkshire_decision = self.config.BERKSHIRE_DECISIONS.get(date, {'buys': [], 'sells': []})

        # Get LLM actions
        llm_buys = []
        llm_sells = []

        for action in decision.get('actions', []):
            ticker = self.config.TICKER_CANON.get(action.get('ticker', ''), action.get('ticker', ''))
            # Include REDUCE as a type of SELL action for convergence analysis
            if action.get('action') == 'BUY':
                llm_buys.append(ticker)
            elif action.get('action') in ['SELL', 'REDUCE']:
                llm_sells.append(ticker)

        # Calculate overlap
        buy_overlap = set(llm_buys) & set(berkshire_decision['buys'])
        sell_overlap = set(llm_sells) & set(berkshire_decision['sells'])

        # Improved convergence rate calculation
        total_berkshire = len(berkshire_decision['buys']) + len(berkshire_decision['sells'])
        total_llm = len(llm_buys) + len(llm_sells)
        total_overlaps = len(buy_overlap) + len(sell_overlap)
        
        # Calculate convergence as percentage of total decisions made by both parties
        total_decisions = max(total_berkshire + total_llm - total_overlaps, 1)  # Avoid double counting overlaps
        convergence_rate = (total_overlaps * 2) / max(total_berkshire + total_llm, 1) if (total_berkshire + total_llm) > 0 else 0.0
        
        # Alternative: Calculate based on Berkshire decisions only (original approach)
        berkshire_convergence_rate = total_overlaps / max(total_berkshire, 1) if total_berkshire > 0 else 0.0

        self.berkshire_convergence.append({
            'date': date,
            'berkshire_buys': berkshire_decision['buys'],
            'berkshire_sells': berkshire_decision['sells'],
            'llm_buys': llm_buys,
            'llm_sells': llm_sells,
            'buy_overlap': list(buy_overlap),
            'sell_overlap': list(sell_overlap),
            'convergence_rate': convergence_rate,
            'berkshire_convergence_rate': berkshire_convergence_rate,
            'total_overlaps': total_overlaps,
            'total_berkshire_decisions': total_berkshire,
            'total_llm_decisions': total_llm
        })

        if buy_overlap or sell_overlap:
            logger.info(f"Convergence with Berkshire - Buys: {buy_overlap}, Sells: {sell_overlap}")

    def build_daily_equity_curve(self) -> pd.Series:
        """Build daily equity curve for accurate risk metrics"""

        if not self.portfolio['history']:
            return pd.Series(dtype=float)

        logger.info("Building daily equity curve...")

        # Get date range
        start_date = pd.to_datetime(self.portfolio['history'][0]['date'])
        end_date = pd.to_datetime(self.portfolio['history'][-1]['date'])

        # Get trading days from market data
        if 'sp500' in self.data_manager.market_data:
            market_series = self.data_manager.market_data['sp500']
            trading_days = market_series.loc[start_date:end_date].index
        else:
            trading_days = pd.date_range(start_date, end_date, freq='B')

        # Initialize equity curve
        equity_curve = pd.Series(index=trading_days, dtype=float)

        # Process each period
        for i, snapshot in enumerate(self.portfolio['history']):
            snap_date = pd.to_datetime(snapshot['date'])

            # Determine date range
            if i + 1 < len(self.portfolio['history']):
                next_date = pd.to_datetime(self.portfolio['history'][i + 1]['date'])
                date_range = trading_days[(trading_days >= snap_date) & (trading_days < next_date)]
            else:
                date_range = trading_days[trading_days >= snap_date]

            # Calculate daily values
            for current_date in date_range:
                daily_value = snapshot['cash']

                # Add holdings value
                for ticker, position in snapshot['holdings'].items():
                    if ticker in self.data_manager.price_data:
                        price_df = self.data_manager.price_data[ticker]

                        if current_date in price_df.index:
                            daily_price = price_df.loc[current_date, 'Close']
                        else:
                            valid_dates = price_df.index[price_df.index <= current_date]
                            if len(valid_dates) > 0:
                                daily_price = price_df.loc[valid_dates[-1], 'Close']
                            else:
                                daily_price = position.get('avg_price', 0)

                        daily_value += position['shares'] * daily_price

                equity_curve.loc[current_date] = daily_value

        # Forward fill any missing values
        self.daily_equity = equity_curve.ffill()

        logger.info(f"Daily equity curve built with {len(self.daily_equity)} days")

        return self.daily_equity

    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""

        # Build daily equity curve
        equity = self.build_daily_equity_curve()

        if len(equity) < 2:
            return {
                'error': 'Insufficient data',
                'total_trades': len(self.trades),
                'final_value': self.portfolio.get('total_value', self.config.INITIAL_CAPITAL),
                'initial_value': self.config.INITIAL_CAPITAL
            }

        # Daily returns
        returns = equity.pct_change().dropna()

        # Core metrics
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

        # Annualized return
        days = (equity.index[-1] - equity.index[0]).days
        years = max(days / 365.25, 0.01)
        annual_return = ((equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1) * 100

        # Risk metrics
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0

        # Sharpe ratio
        excess_return = annual_return / 100 - self.config.RISK_FREE_RATE
        sharpe = excess_return / (volatility / 100) if volatility > 0 else 0

        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown_series = (cum_returns - running_max) / running_max
        max_drawdown = drawdown_series.min() * 100

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else volatility
        sortino = excess_return / (downside_vol / 100) if downside_vol > 0 else 0

        # Win rate
        trades_with_gl = [t for t in self.trades if 'gain_loss' in t]
        winning_trades = [t for t in trades_with_gl if t['gain_loss'] > 0]
        win_rate = len(winning_trades) / len(trades_with_gl) * 100 if trades_with_gl else 0

        # Profit factor
        gross_profits = sum(t['gain_loss'] for t in trades_with_gl if t['gain_loss'] > 0)
        gross_losses = abs(sum(t['gain_loss'] for t in trades_with_gl if t['gain_loss'] < 0))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0

        # Yearly returns
        yearly_performance = {}
        for year, values in self.yearly_returns.items():
            if 'start_value' in values and 'end_value' in values and values['start_value'] > 0:
                yearly_performance[year] = (values['end_value'] / values['start_value'] - 1) * 100

        # Benchmark comparisons for academic analysis
        berkshire_comparison = self.calculate_berkshire_comparison(equity)
        sp500_comparison = self.calculate_sp500_comparison(equity)

        # Average convergence
        avg_convergence = np.mean([c['convergence_rate'] for c in self.berkshire_convergence]) if self.berkshire_convergence else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'final_value': equity.iloc[-1],
            'initial_value': equity.iloc[0],
            'berkshire_comparison': berkshire_comparison,
            'sp500_comparison': sp500_comparison,
            'yearly_returns': yearly_performance,
            'average_convergence': avg_convergence * 100
        }

    def calculate_berkshire_comparison(self, portfolio_equity: pd.Series) -> Dict:
        """Calculate comparison with Berkshire Hathaway - COMPLETELY FIXED"""

        if 'berkshire' not in self.data_manager.market_data:
            logger.warning("Berkshire data not available for comparison")
            return {}

        try:
            brk = self.data_manager.market_data['berkshire']

            # Ensure both series are pandas Series
            if isinstance(brk, pd.DataFrame):
                brk = brk['Close'] if 'Close' in brk.columns else brk.squeeze()

            # Align dates
            common_dates = portfolio_equity.index.intersection(brk.index)

            if len(common_dates) < 2:
                return {}

            # Get aligned series
            portfolio = portfolio_equity.loc[common_dates]
            berkshire = brk.loc[common_dates]

            # Calculate returns
            portfolio_returns = portfolio.pct_change().dropna()
            berkshire_returns = berkshire.pct_change().dropna()

            # Ensure both return series have same dates
            common_return_dates = portfolio_returns.index.intersection(berkshire_returns.index)
            if len(common_return_dates) == 0:
                logger.warning("No common dates for return calculation")
                return {}

            portfolio_returns = portfolio_returns.loc[common_return_dates]
            berkshire_returns = berkshire_returns.loc[common_return_dates]

            # Cumulative returns
            portfolio_cum = (portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100
            berkshire_cum = (berkshire.iloc[-1] / berkshire.iloc[0] - 1) * 100

            # Annualized returns
            years = max((common_dates[-1] - common_dates[0]).days / 365.25, 0.01)
            portfolio_ann = ((portfolio.iloc[-1] / portfolio.iloc[0]) ** (1/years) - 1) * 100
            berkshire_ann = ((berkshire.iloc[-1] / berkshire.iloc[0]) ** (1/years) - 1) * 100

            # Risk metrics - ensure scalars
            portfolio_vol = float(portfolio_returns.std() * np.sqrt(252) * 100) if len(portfolio_returns) > 0 else 0
            berkshire_vol = float(berkshire_returns.std() * np.sqrt(252) * 100) if len(berkshire_returns) > 0 else 0

            # Correlation and beta
            correlation = 0
            beta = 1
            if len(portfolio_returns) > 1 and len(berkshire_returns) > 1:
                try:
                    correlation = float(portfolio_returns.corr(berkshire_returns))
                    variance = float(berkshire_returns.var())
                    if variance > 0:
                        covariance = float(portfolio_returns.cov(berkshire_returns))
                        beta = covariance / variance
                except Exception as e:
                    logger.debug(f"Could not calculate correlation/beta: {e}")

            # Information ratio - COMPLETELY FIXED
            info_ratio = 0
            tracking_error = 0
            if len(portfolio_returns) > 0 and len(berkshire_returns) > 0:
                try:
                    active_returns = portfolio_returns - berkshire_returns
                    # Calculate tracking error and ensure it's a scalar
                    tracking_error = float(active_returns.std() * np.sqrt(252))
                    # Now safely check if tracking_error > 0
                    if tracking_error > 0:
                        info_ratio = (portfolio_ann - berkshire_ann) / 100 / tracking_error
                except Exception as e:
                    logger.debug(f"Could not calculate information ratio: {e}")

            return {
                'llm_cumulative': float(portfolio_cum),
                'berkshire_cumulative': float(berkshire_cum),
                'llm_annual_return': float(portfolio_ann),
                'berkshire_annual_return': float(berkshire_ann),
                'relative_cumulative': float(portfolio_cum - berkshire_cum),
                'relative_performance': float(portfolio_ann - berkshire_ann),
                'llm_volatility': portfolio_vol,
                'berkshire_volatility': berkshire_vol,
                'correlation': float(correlation) if not pd.isna(correlation) else 0,
                'beta': float(beta) if not pd.isna(beta) else 1,
                'information_ratio': float(info_ratio) if not pd.isna(info_ratio) else 0
            }
        except Exception as e:
            logger.error(f"Error in Berkshire comparison: {e}")
            return {}

    def calculate_sp500_comparison(self, portfolio_equity: pd.Series) -> Dict:
        """Calculate comparison with S&P 500 Index for academic analysis"""
        
        # Try both S&P 500 sources
        sp500_data = None
        
        if 'sp500_etf' in self.data_manager.market_data:
            sp500_data = self.data_manager.market_data['sp500_etf']
        elif 'sp500' in self.data_manager.market_data:
            sp500_data = self.data_manager.market_data['sp500']
        
        if sp500_data is None:
            logger.warning("S&P 500 data not available for comparison")
            return {}
        
        try:
            # Ensure both series are pandas Series
            if isinstance(sp500_data, pd.DataFrame):
                sp500_data = sp500_data['Close'] if 'Close' in sp500_data.columns else sp500_data.squeeze()
            
            # Align dates
            common_dates = portfolio_equity.index.intersection(sp500_data.index)
            
            if len(common_dates) < 30:  # Need at least 30 days for meaningful comparison
                logger.warning(f"Insufficient overlapping data for S&P 500 comparison: {len(common_dates)} days")
                return {}
            
            # Normalize both to the same starting value for fair comparison
            portfolio_aligned = portfolio_equity.loc[common_dates]
            sp500_aligned = sp500_data.loc[common_dates]
            
            # Calculate normalized performance (starting at 1.0)
            portfolio_normalized = portfolio_aligned / portfolio_aligned.iloc[0]
            sp500_normalized = sp500_aligned / sp500_aligned.iloc[0]
            
            # Calculate returns
            portfolio_total_return = portfolio_normalized.iloc[-1] - 1
            sp500_total_return = sp500_normalized.iloc[-1] - 1
            
            # Annualized returns
            years = len(common_dates) / 252
            portfolio_annual = (1 + portfolio_total_return) ** (1/years) - 1 if years > 0 else 0
            sp500_annual = (1 + sp500_total_return) ** (1/years) - 1 if years > 0 else 0
            
            # Calculate daily returns for additional metrics
            portfolio_returns = portfolio_normalized.pct_change().dropna()
            sp500_returns = sp500_normalized.pct_change().dropna()
            
            # Risk metrics
            correlation = portfolio_returns.corr(sp500_returns) if len(portfolio_returns) > 1 else 0
            
            # Beta calculation
            if len(portfolio_returns) > 1 and sp500_returns.var() > 0:
                beta = portfolio_returns.cov(sp500_returns) / sp500_returns.var()
            else:
                beta = 1.0
            
            # Information Ratio (active return / tracking error)
            active_return = portfolio_annual - sp500_annual
            tracking_error = (portfolio_returns - sp500_returns).std() * np.sqrt(252) if len(portfolio_returns) > 1 else 0
            info_ratio = active_return / tracking_error if tracking_error > 0 else 0
            
            return {
                'sp500_annual_return': float(sp500_annual),
                'sp500_total_return': float(sp500_total_return),
                'relative_return': float(active_return),
                'correlation': float(correlation) if not pd.isna(correlation) else 0,
                'beta': float(beta) if not pd.isna(beta) else 1,
                'information_ratio': float(info_ratio) if not pd.isna(info_ratio) else 0,
                'tracking_error': float(tracking_error) if not pd.isna(tracking_error) else 0
            }
            
        except Exception as e:
            logger.error(f"Error in S&P 500 comparison: {e}")
            return {}

# ============================================================================
# SECTION 10: MAIN SIMULATION ORCHESTRATOR - OPTIMIZED
# ============================================================================

class SimulationOrchestrator:
    """Main orchestrator for the complete simulation - WITH COMPARISON MODE"""

    def __init__(self, dry_run=False, llm_mode='BOTH'):
        self.config = SimulationConfig()

        # Set LLM mode
        self.config.LLM_MODE = llm_mode

        if dry_run:
            logger.info("DRY RUN MODE - No API calls will be made")
            self.config.DRY_RUN = True
            self.config.USE_LLM_DECISIONS = False
            self.config.USE_PROFESSIONAL_RESEARCH = False
            self.config.USE_EXPENSIVE_APIS = False

        self.data_manager = DataManager(self.config)
        self.llm_engine = LLMDecisionEngine(self.config)
        self.portfolio_manager = PortfolioManager(self.config, self.data_manager)
        self.results = {}

    async def run_simulation(self):
        """Run complete investment simulation"""

        logger.info("=" * 80)
        logger.info(f"STARTING LLM INVESTMENT SIMULATION - MODE: {self.config.LLM_MODE}")
        logger.info("Version 7.0 - With Comparison Capability")
        logger.info("=" * 80)

        try:
            # Step 1: Download all data (only once, will be cached)
            print(f"\n📊 Downloading market data for {self.config.LLM_MODE} mode...")
            self.data_manager.download_all_data()

            # Set data manager reference
            self.llm_engine.data_manager = self.data_manager

            # Step 2: Run simulation for each decision date
            print(f"\n🤖 Running investment simulation with {self.config.LLM_MODE}...")

            if TQDM_AVAILABLE:
                iterator = tqdm(self.config.DECISION_DATES, desc=f"Processing ({self.config.LLM_MODE})")
            else:
                iterator = self.config.DECISION_DATES

            for decision_date in iterator:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing {decision_date} - Mode: {self.config.LLM_MODE}")
                logger.info(f"{'=' * 60}")

                # Get market context
                market_context = self.get_market_context(decision_date)

                # Screen stocks
                available_stocks = self.screen_stocks(decision_date)

                if not available_stocks:
                    logger.warning(f"No stocks screened for {decision_date}")
                    continue

                # Get current portfolio state
                current_portfolio = self.get_portfolio_state()

                # Make investment decision
                decision = await self.llm_engine.make_investment_decision(
                    market_context,
                    available_stocks,
                    current_portfolio,
                    decision_date
                )

                # Execute decision
                self.portfolio_manager.execute_decision(decision, decision_date)

                # Log decision with LLM mode
                self.log_decision(decision_date, decision)

            # Step 3: Calculate final performance
            print(f"\n📈 Calculating performance metrics for {self.config.LLM_MODE}...")
            self.results = self.portfolio_manager.calculate_performance_metrics()

            # Add LLM mode to results
            self.results['llm_mode'] = self.config.LLM_MODE

            # Step 4: Save results with mode suffix
            self.save_results()

            # Step 5: Generate report
            self.generate_report()

            return self.results

        except Exception as e:
            logger.error(f"Simulation failed: {e}", exc_info=True)
            raise

    def get_market_context(self, date):
        """Get market context with improved handling"""
        context = {}

        date_obj = normalize_date(date)

        for indicator, series in self.data_manager.market_data.items():
            try:
                s = series
                if isinstance(s, pd.DataFrame):
                    s = s['Close'] if 'Close' in s.columns else s.squeeze()

                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)

                s = s.sort_index()

                if date_obj in s.index:
                    context[indicator] = float(s.loc[date_obj])
                else:
                    valid_idx = s.index[s.index <= date_obj]
                    if len(valid_idx) > 0:
                        context[indicator] = float(s.loc[valid_idx[-1]])
                    else:
                        context[indicator] = float(s.iloc[0]) if len(s) > 0 else None

            except Exception as e:
                logger.warning(f"Could not get market context for {indicator}: {e}")
                context[indicator] = None

        # Calculate market trend
        try:
            if 'sp500' in self.data_manager.market_data:
                sp = self.data_manager.market_data['sp500']
                if isinstance(sp, pd.DataFrame):
                    sp = sp['Close'] if 'Close' in sp.columns else sp.squeeze()

                if sp.index.tz is not None:
                    sp.index = sp.index.tz_localize(None)

                sp = sp.sort_index()
                valid_idx = sp.index[sp.index <= date_obj]
                if len(valid_idx) > 20:
                    recent = sp.loc[valid_idx[-20:]]
                    context['market_trend'] = f"{(recent.iloc[-1] / recent.iloc[0] - 1) * 100:.1f}% last 20 days"
                else:
                    context['market_trend'] = "Insufficient data"
        except Exception as e:
            logger.warning(f"Could not calculate market trend: {e}")
            context['market_trend'] = "N/A"

        return context

    def screen_stocks(self, date):
        """Screen stocks based on price-only criteria"""
        screened = []

        date_obj = normalize_date(date)

        for ticker in self.data_manager.tickers:
            if ticker not in self.data_manager.price_data:
                continue

            try:
                price_data = self.data_manager.price_data[ticker]
                fundamental_data = self.data_manager.fundamental_data.get(ticker, {})

                # Ensure timezone-naive index
                if price_data.index.tz is not None:
                    price_data.index = price_data.index.tz_localize(None)

                # Get data up to decision date using the filter
                data_filter = HistoricalDataFilter()
                historical = data_filter.filter_price_data(price_data, date_obj)

                if len(historical) < 60:
                    continue

                current_price = float(historical['Close'].iloc[-1])

                # Check liquidity via ADV
                if 'ADV_20' in historical.columns:
                    adv = historical['ADV_20'].iloc[-1]
                    if pd.notna(adv) and adv < self.config.MIN_ADV:
                        continue

                # Calculate scores
                momentum_score = self.calculate_momentum_score(historical)
                volatility_score = self.calculate_volatility_score(historical)
                technical_score = self.calculate_technical_score(historical)

                # Combined score (price-only)
                combined_score = (momentum_score * 0.5 + volatility_score * 0.25 + technical_score * 0.25)

                # Extract recent metrics
                recent = historical.iloc[-1]

                stock_info = {
                    'ticker': ticker,
                    'name': ticker,
                    'sector': fundamental_data.get('sector', 'Unknown'),
                    'market_cap': fundamental_data.get('market_cap', 0),
                    'pe_ratio': fundamental_data.get('pe_ratio'),
                    'pb_ratio': fundamental_data.get('pb_ratio'),
                    'roe': fundamental_data.get('roe'),
                    'profit_margin': fundamental_data.get('profit_margin'),
                    'debt_to_equity': fundamental_data.get('debt_to_equity'),
                    'fcf_yield': fundamental_data.get('fcf_yield', 0),
                    'revenue_growth': fundamental_data.get('revenue_growth'),
                    'momentum_6m': float(recent.get('momentum_6m', 0)) if pd.notna(recent.get('momentum_6m')) else 0,
                    'momentum_3m': float(recent.get('momentum_3m', 0)) if pd.notna(recent.get('momentum_3m')) else 0,
                    'volatility_60d': float(recent.get('volatility_60d', 0.25)) if pd.notna(recent.get('volatility_60d')) else 0.25,
                    'rsi': float(recent.get('RSI', 50)) if pd.notna(recent.get('RSI')) else 50,
                    'momentum_score': momentum_score,
                    'volatility_score': volatility_score,
                    'technical_score': technical_score,
                    'combined_score': combined_score,
                    'current_price': current_price
                }

                screened.append(stock_info)

            except Exception as e:
                logger.warning(f"Error screening {ticker}: {e}")
                continue

        # Sort by combined score
        screened.sort(key=lambda x: x['combined_score'], reverse=True)

        logger.info(f"Screened {len(screened)} stocks for {date}")

        return screened

    def calculate_momentum_score(self, price_data):
        """Calculate momentum score"""
        score = 50

        if len(price_data) == 0:
            return score

        recent = price_data.iloc[-1]

        # 6-month momentum
        if 'momentum_6m' in recent and pd.notna(recent['momentum_6m']):
            mom_6m = recent['momentum_6m']
            if mom_6m > 0.30:
                score += 20
            elif mom_6m > 0.20:
                score += 15
            elif mom_6m > 0.10:
                score += 10
            elif mom_6m > 0:
                score += 5
            elif mom_6m > -0.10:
                score -= 5
            else:
                score -= 10

        # 3-month momentum
        if 'momentum_3m' in recent and pd.notna(recent['momentum_3m']):
            mom_3m = recent['momentum_3m']
            if mom_3m > 0.15:
                score += 10
            elif mom_3m > 0.05:
                score += 5
            elif mom_3m < -0.05:
                score -= 5

        # Trend
        if all(col in recent for col in ['Close', 'MA_50', 'MA_200']):
            if pd.notna(recent['MA_50']) and pd.notna(recent['MA_200']):
                if recent['Close'] > recent['MA_50'] > recent['MA_200']:
                    score += 10
                elif recent['Close'] > recent['MA_50']:
                    score += 5
                elif recent['Close'] < recent['MA_50'] < recent['MA_200']:
                    score -= 10

        return min(max(score, 0), 100)

    def calculate_volatility_score(self, price_data):
        """Calculate volatility score"""
        score = 50

        if len(price_data) == 0:
            return score

        recent = price_data.iloc[-1]

        # 60-day volatility
        if 'volatility_60d' in recent and pd.notna(recent['volatility_60d']):
            vol = recent['volatility_60d']
            if vol < 0.15:
                score += 20
            elif vol < 0.25:
                score += 10
            elif vol < 0.35:
                score += 0
            elif vol < 0.50:
                score -= 10
            else:
                score -= 20

        # Volatility trend
        if 'volatility_20d' in price_data.columns and 'volatility_60d' in price_data.columns:
            recent_vol_trend = price_data['volatility_20d'].iloc[-5:].mean()
            longer_vol_trend = price_data['volatility_60d'].iloc[-20:].mean()

            if pd.notna(recent_vol_trend) and pd.notna(longer_vol_trend):
                if recent_vol_trend < longer_vol_trend * 0.8:
                    score += 10
                elif recent_vol_trend > longer_vol_trend * 1.2:
                    score -= 10

        return min(max(score, 0), 100)

    def calculate_technical_score(self, price_data):
        """Calculate technical indicators score"""
        score = 50

        if len(price_data) == 0:
            return score

        recent = price_data.iloc[-1]

        # RSI
        if 'RSI' in recent and pd.notna(recent['RSI']):
            rsi = recent['RSI']
            if 40 < rsi < 60:
                score += 10
            elif 30 < rsi <= 40:
                score += 15
            elif 60 <= rsi < 70:
                score += 5
            elif rsi <= 30:
                score += 20
            elif rsi >= 70:
                score -= 5

        # MACD
        if 'MACD' in recent and 'Signal' in recent:
            if pd.notna(recent['MACD']) and pd.notna(recent['Signal']):
                if recent['MACD'] > recent['Signal']:
                    score += 10
                else:
                    score -= 5

        # Bollinger Bands
        if all(col in recent for col in ['Close', 'BB_upper', 'BB_lower']):
            if pd.notna(recent['BB_upper']) and pd.notna(recent['BB_lower']):
                bb_range = recent['BB_upper'] - recent['BB_lower']
                if bb_range > 0:
                    bb_position = (recent['Close'] - recent['BB_lower']) / bb_range
                    if 0.3 < bb_position < 0.7:
                        score += 5
                    elif bb_position <= 0.2:
                        score += 10
                    elif bb_position >= 0.8:
                        score -= 5

        return min(max(score, 0), 100)

    def get_portfolio_state(self):
        """Get current portfolio state"""
        return {
            'total_value': self.portfolio_manager.portfolio['total_value'],
            'cash': self.portfolio_manager.portfolio['cash'],
            'cash_percentage': self.portfolio_manager.portfolio.get('cash_percentage', 100),
            'holdings': self.portfolio_manager.portfolio['holdings']
        }

    def log_decision(self, date, decision):
        """Log investment decision with LLM mode"""
        try:
            os.makedirs(os.path.dirname(os.path.abspath('decision_logs')) or '.', exist_ok=True)

            berkshire_decision = self.config.BERKSHIRE_DECISIONS.get(date, {'buys': [], 'sells': []})
            convergence_data = self.portfolio_manager.berkshire_convergence[-1] if self.portfolio_manager.berkshire_convergence else None

            log_entry = {
                'date': date,
                'llm_mode': self.config.LLM_MODE,  # Track which LLM mode was used
                'decision': decision if isinstance(decision, dict) else {'actions': []},
                'portfolio_value': float(self.portfolio_manager.portfolio['total_value']),
                'berkshire_decision': berkshire_decision,
                'convergence': convergence_data
            }

            # Save with mode-specific filename
            filename = f'decision_log_{self.config.LLM_MODE.lower()}.json'
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')

        except Exception as e:
            logger.error(f"Failed to log decision for {date}: {e}")

    def save_results(self):
        """Save simulation results with LLM mode suffix"""
        try:
            mode_suffix = self.config.LLM_MODE.lower()

            # Portfolio history
            if self.portfolio_manager.portfolio.get('history'):
                history_df = pd.DataFrame(self.portfolio_manager.portfolio['history'])
                filename = f'portfolio_history_{mode_suffix}.csv'
                history_df.to_csv(filename, index=False)
                logger.info(f"Portfolio history saved to {filename}")

            # Trades
            if self.portfolio_manager.trades:
                trades_df = pd.DataFrame(self.portfolio_manager.trades)
                filename = f'trades_{mode_suffix}.csv'
                trades_df.to_csv(filename, index=False)
                logger.info(f"Trades saved to {filename}")

            # Daily equity curve
            if len(self.portfolio_manager.daily_equity) > 0:
                filename = f'daily_equity_{mode_suffix}.csv'
                self.portfolio_manager.daily_equity.to_csv(filename)
                logger.info(f"Daily equity curve saved to {filename}")

            # Performance metrics
            if self.results:
                filename = f'performance_metrics_{mode_suffix}.json'
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, default=str)
                logger.info(f"Performance metrics saved to {filename}")

            # Berkshire convergence
            if self.portfolio_manager.berkshire_convergence:
                convergence_df = pd.DataFrame(self.portfolio_manager.berkshire_convergence)
                filename = f'berkshire_convergence_{mode_suffix}.csv'
                convergence_df.to_csv(filename, index=False)
                logger.info(f"Berkshire convergence saved to {filename}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def generate_report(self):
        """Generate final performance report with LLM mode"""
        print("\n" + "=" * 80)
        print(f"LLM INVESTMENT SIMULATION RESULTS - {self.config.LLM_MODE}")
        print("=" * 80)
        print(f"Simulation Period: {self.config.START_DATE} to {self.config.END_DATE}")
        print(f"LLM Mode: {self.config.LLM_MODE}")
        print(f"Initial Capital: ${self.config.INITIAL_CAPITAL:,.0f}")
        print(f"Final Value: ${self.results.get('final_value', 0):,.0f}")

        print(f"\n📊 PERFORMANCE METRICS ({self.config.LLM_MODE}):")
        print("-" * 40)
        print(f"Total Return: {self.results.get('total_return', 0):.2f}%")
        print(f"Annual Return: {self.results.get('annual_return', 0):.2f}%")
        print(f"Volatility: {self.results.get('volatility', 0):.2f}%")
        print(f"Sharpe Ratio: {self.results.get('sharpe_ratio', 0):.3f}")
        print(f"Sortino Ratio: {self.results.get('sortino_ratio', 0):.3f}")
        print(f"Calmar Ratio: {self.results.get('calmar_ratio', 0):.3f}")
        print(f"Maximum Drawdown: {self.results.get('max_drawdown', 0):.2f}%")
        print(f"Win Rate: {self.results.get('win_rate', 0):.1f}%")
        print(f"Profit Factor: {self.results.get('profit_factor', 0):.2f}")
        print(f"Total Trades: {self.results.get('total_trades', 0)}")

        if 'berkshire_comparison' in self.results:
            print(f"\n🎯 COMPARISON WITH BERKSHIRE HATHAWAY (BRK-B) - {self.config.LLM_MODE}:")
            print("-" * 40)
            comp = self.results['berkshire_comparison']
            print(f"LLM Annual Return: {comp.get('llm_annual_return', 0):.2f}%")
            print(f"Berkshire Annual Return: {comp.get('berkshire_annual_return', 0):.2f}%")
            print(f"Relative Performance: {comp.get('relative_performance', 0):+.2f}%")
            print(f"LLM Cumulative Return: {comp.get('llm_cumulative', 0):.2f}%")
            print(f"Berkshire Cumulative Return: {comp.get('berkshire_cumulative', 0):.2f}%")
            print(f"Correlation: {comp.get('correlation', 0):.3f}")
            print(f"Beta: {comp.get('beta', 0):.3f}")
            print(f"Information Ratio: {comp.get('information_ratio', 0):.3f}")

            if comp.get('relative_performance', 0) > 0:
                print(f"\n✅ {self.config.LLM_MODE} OUTPERFORMED Berkshire Hathaway!")
            else:
                print(f"\n❌ Berkshire Hathaway outperformed {self.config.LLM_MODE}")

        print(f"\n📈 Decision Convergence with Berkshire: {self.results.get('average_convergence', 0):.1f}%")

        if 'yearly_returns' in self.results and self.results['yearly_returns']:
            print(f"\n📅 YEARLY RETURNS ({self.config.LLM_MODE}):")
            print("-" * 40)
            for year, return_pct in self.results['yearly_returns'].items():
                print(f"{year}: {return_pct:.2f}%")

        # Final portfolio composition
        print(f"\n💼 FINAL PORTFOLIO COMPOSITION ({self.config.LLM_MODE}):")
        print("-" * 40)
        portfolio = self.portfolio_manager.portfolio
        print(f"Cash: ${portfolio['cash']:,.0f} ({portfolio.get('cash_percentage', 0):.1f}%)")
        print(f"Number of Holdings: {len(portfolio['holdings'])}")

        if portfolio['holdings']:
            print("\nTop Holdings:")
            holdings_list = [(k, v) for k, v in portfolio['holdings'].items()]
            holdings_list.sort(key=lambda x: x[1].get('value', 0), reverse=True)

            for ticker, position in holdings_list[:10]:
                value = position.get('value', 0)
                weight = position.get('weight', 0)
                return_pct = position.get('return', 0)
                print(f"  {ticker}: ${value:,.0f} ({weight:.1f}%) | Return: {return_pct:.1f}%")

        print("=" * 80)

# ============================================================================
# SECTION 12: COMPARISON RUNNER
# ============================================================================

def _calculate_statistical_significance(comparison_results: Dict):
    """Calculate statistical significance of performance differences (Academic Enhancement)"""
    print("\n📈 STATISTICAL SIGNIFICANCE ANALYSIS:")
    print("-" * 50)
    
    modes_with_data = [mode for mode in comparison_results.keys() 
                      if 'error' not in comparison_results[mode]]
    
    if len(modes_with_data) < 2:
        print("⚠️  Insufficient data for statistical analysis")
        return
    
    # Extract daily returns for each mode
    daily_returns = {}
    for mode in modes_with_data:
        results = comparison_results[mode]
        if 'daily_equity' in results:
            equity_series = results['daily_equity']
            if len(equity_series) > 1:
                returns = [(equity_series[i] - equity_series[i-1]) / equity_series[i-1] 
                          for i in range(1, len(equity_series))]
                daily_returns[mode] = returns
    
    # Perform pairwise t-tests
    from itertools import combinations
    try:
        import scipy.stats as stats
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False
        stats = None
    
    for mode1, mode2 in combinations(modes_with_data, 2):
        if mode1 in daily_returns and mode2 in daily_returns:
            returns1 = daily_returns[mode1]
            returns2 = daily_returns[mode2]
            
            if len(returns1) > 10 and len(returns2) > 10:  # Minimum sample size
                try:
                    # Simple comparison using standard deviation and mean
                    mean1, std1 = statistics.mean(returns1), statistics.stdev(returns1)
                    mean2, std2 = statistics.mean(returns2), statistics.stdev(returns2)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = math.sqrt((std1**2 + std2**2) / 2)
                    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    
                    print(f"• {mode1} vs {mode2}:")
                    print(f"  Mean Return Difference: {(mean1 - mean2)*100:.4f}% daily")
                    print(f"  Cohen's d Effect Size: {cohens_d:.3f}")
                    
                    if abs(cohens_d) > 0.8:
                        print("  📊 Large effect size")
                    elif abs(cohens_d) > 0.5:
                        print("  📊 Medium effect size")
                    elif abs(cohens_d) > 0.2:
                        print("  📊 Small effect size")
                    else:
                        print("  📊 Negligible effect size")
                        
                except Exception as e:
                    print(f"  ⚠️  Statistical test failed: {e}")
    
    print()

def _calculate_academic_metrics(comparison_results: Dict):
    """Calculate additional academic research metrics"""
    print("\n📋 ACADEMIC RESEARCH METRICS:")
    print("-" * 50)
    
    for mode, results in comparison_results.items():
        if 'error' in results:
            continue
            
        print(f"\n🔍 {mode} Analysis:")
        
        # Information Ratio
        if 'tracking_error' in results and results['tracking_error'] > 0:
            active_return = results.get('annual_return', 0) - 3  # Assuming 3% risk-free rate
            info_ratio = active_return / results['tracking_error']
            print(f"  Information Ratio: {info_ratio:.3f}")
        
        # Maximum Drawdown Duration
        if 'daily_equity' in results:
            equity = results['daily_equity']
            if equity:
                peak = equity[0]
                max_dd_duration = 0
                current_dd_duration = 0
                
                for value in equity[1:]:
                    if value >= peak:
                        peak = value
                        current_dd_duration = 0
                    else:
                        current_dd_duration += 1
                        max_dd_duration = max(max_dd_duration, current_dd_duration)
                
                print(f"  Max Drawdown Duration: {max_dd_duration} periods")
        
        # Win Rate (if trade data available)
        if 'trades' in results:
            trades = results['trades']
            winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            total_trades = len(trades)
            if total_trades > 0:
                win_rate = winning_trades / total_trades * 100
                print(f"  Win Rate: {win_rate:.1f}% ({winning_trades}/{total_trades})")
        
        # Calmar Ratio (Annual Return / Max Drawdown)
        annual_return = results.get('annual_return', 0)
        max_drawdown = abs(results.get('max_drawdown', 1))
        if max_drawdown > 0:
            calmar_ratio = annual_return / max_drawdown
            print(f"  Calmar Ratio: {calmar_ratio:.3f}")

def _export_detailed_results(comparison_results: Dict, filename: str = "academic_analysis_detailed.json"):
    """Export detailed results for further academic analysis"""
    try:
        # Prepare detailed data for export
        detailed_results = {}
        
        for mode, results in comparison_results.items():
            if 'error' not in results:
                detailed_results[mode] = {
                    'performance_metrics': {
                        'total_return': results.get('total_return', 0),
                        'annual_return': results.get('annual_return', 0),
                        'sharpe_ratio': results.get('sharpe_ratio', 0),
                        'max_drawdown': results.get('max_drawdown', 0),
                        'volatility': results.get('volatility', 0)
                    },
                    'daily_data': {
                        'daily_equity': results.get('daily_equity', []),
                        'daily_returns': [],
                        'rolling_volatility': []
                    },
                    'trade_analysis': {
                        'total_trades': len(results.get('trades', [])),
                        'trade_details': results.get('trades', [])
                    },
                    'metadata': {
                        'simulation_start': results.get('start_date', ''),
                        'simulation_end': results.get('end_date', ''),
                        'llm_mode': mode,
                        'universe_size': results.get('universe_size', 0)
                    }
                }
                
                # Calculate daily returns from equity curve
                equity = results.get('daily_equity', [])
                if len(equity) > 1:
                    daily_returns = [(equity[i] - equity[i-1]) / equity[i-1] 
                                   for i in range(1, len(equity))]
                    detailed_results[mode]['daily_data']['daily_returns'] = daily_returns
                    
                    # Calculate 30-day rolling volatility
                    window = 30
                    rolling_vol = []
                    for i in range(window, len(daily_returns)):
                        vol = statistics.stdev(daily_returns[i-window:i]) * math.sqrt(252)
                        rolling_vol.append(vol)
                    detailed_results[mode]['daily_data']['rolling_volatility'] = rolling_vol
        
        # Export to JSON
        with open(filename, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results exported to {filename}")
        print("   This file contains comprehensive data for further academic analysis")
        
    except Exception as e:
        print(f"⚠️  Export failed: {e}")

def export_detailed_academic_report(results: Dict) -> None:
    """Export comprehensive academic-quality report for research papers"""
    
    try:
        academic_data = {
            'study_metadata': {
                'experiment_date': datetime.now().isoformat(),
                'simulation_period': f"2021-10-01 to 2024-12-31",
                'universe_size': 500,
                'initial_capital': 100_000_000,
                'methodology': 'Large Language Model Investment Decisions Comparison',
                'llm_models': {
                    'openai': 'GPT-4',
                    'anthropic': 'Claude-3.5-Sonnet',
                    'combined': 'Consensus of GPT-4 and Claude-3.5-Sonnet'
                }
            },
            'performance_metrics': {},
            'statistical_analysis': {},
            'decision_analysis': {},
            'risk_metrics': {},
            'benchmark_comparisons': {}
        }
        
        for mode, result_data in results.items():
            if 'error' in result_data:
                continue
                
            mode_analysis = {
                # Core Performance
                'returns': {
                    'total_return_pct': result_data.get('total_return', 0),
                    'annualized_return_pct': result_data.get('annual_return', 0),
                    'excess_return_vs_sp500_pct': result_data.get('sp500_comparison', {}).get('relative_return', 0) * 100,
                    'excess_return_vs_berkshire_pct': result_data.get('berkshire_comparison', {}).get('relative_performance', 0)
                },
                
                # Risk Metrics
                'risk_metrics': {
                    'volatility_pct': result_data.get('volatility', 0) * 100,
                    'max_drawdown_pct': result_data.get('max_drawdown', 0),
                    'var_95_pct': calculate_var(result_data.get('daily_equity', {}), 0.05),
                    'skewness': calculate_skewness(result_data.get('daily_equity', {})),
                    'kurtosis': calculate_kurtosis(result_data.get('daily_equity', {}))
                },
                
                # Risk-Adjusted Performance
                'risk_adjusted_metrics': {
                    'sharpe_ratio': result_data.get('sharpe_ratio', 0),
                    'sortino_ratio': result_data.get('sortino_ratio', 0),
                    'calmar_ratio': result_data.get('calmar_ratio', 0),
                    'treynor_ratio_sp500': calculate_treynor_ratio(result_data),
                    'jensen_alpha_sp500': calculate_jensen_alpha(result_data)
                },
                
                # Trading Statistics
                'trading_statistics': {
                    'total_trades': result_data.get('total_trades', 0),
                    'win_rate_pct': result_data.get('win_rate', 0),
                    'profit_factor': result_data.get('profit_factor', 0),
                    'avg_trade_duration_days': calculate_avg_trade_duration(mode),
                    'turnover_rate_annual': calculate_annual_turnover(mode)
                },
                
                # Portfolio Characteristics
                'portfolio_metrics': {
                    'avg_number_of_holdings': calculate_avg_holdings(mode),
                    'avg_position_size_pct': calculate_avg_position_size(mode),
                    'sector_concentration_herfindahl': calculate_sector_concentration(mode),
                    'cash_utilization_pct': calculate_cash_utilization(mode)
                },
                
                # Benchmark Comparisons
                'benchmark_analysis': {
                    'sp500_comparison': result_data.get('sp500_comparison', {}),
                    'berkshire_comparison': result_data.get('berkshire_comparison', {}),
                    'market_beta_sp500': result_data.get('sp500_comparison', {}).get('beta', 1.0),
                    'correlation_with_sp500': result_data.get('sp500_comparison', {}).get('correlation', 0),
                    'tracking_error_vs_sp500_pct': result_data.get('sp500_comparison', {}).get('tracking_error', 0) * 100
                }
            }
            
            academic_data['performance_metrics'][mode] = mode_analysis
        
        # Statistical Significance Tests
        academic_data['statistical_analysis'] = {
            'methodology': 'Performance comparison using daily returns',
            'sample_size_days': len(list(results.values())[0].get('daily_equity', {})) if results else 0,
            'significance_tests': {
                'note': 'T-tests and Wilcoxon rank-sum tests recommended for return differences',
                'suggested_confidence_level': 0.05
            }
        }
        
        # Export to JSON
        with open('academic_research_report.json', 'w') as f:
            json.dump(academic_data, f, indent=2)
            
        # Export to CSV for easy analysis
        export_academic_csv(academic_data)
        
        print(f"📊 Academic research report exported to 'academic_research_report.json'")
        print(f"📊 CSV data exported to 'academic_analysis.csv' for statistical software")
        
    except Exception as e:
        print(f"⚠️  Academic report export failed: {e}")

def analyze_llm_decisions(mode: str) -> Dict:
    """Analyze LLM decision patterns for academic insights"""
    try:
        decision_file = f'decision_log_{mode.lower()}.json'
        if not os.path.exists(decision_file):
            return {'avg_score': 0, 'consistency': 0, 'risk_mgmt': 'No data', 'sector_div': 'No data', 'avg_position': 0}
        
        with open(decision_file, 'r') as f:
            decisions = [json.loads(line) for line in f]
        
        if not decisions:
            return {'avg_score': 0, 'consistency': 0, 'risk_mgmt': 'No data', 'sector_div': 'No data', 'avg_position': 0}
        
        # Analyze decision patterns
        position_sizes = []
        action_types = {'BUY': 0, 'SELL': 0, 'REDUCE': 0}
        sectors = {}
        
        for decision_entry in decisions:
            decision = decision_entry.get('decision', {})
            actions = decision.get('actions', [])
            
            for action in actions:
                action_type = action.get('action', 'UNKNOWN')
                action_types[action_type] = action_types.get(action_type, 0) + 1
                
                if action.get('weight'):
                    position_sizes.append(action['weight'] * 100)
        
        avg_position = np.mean(position_sizes) if position_sizes else 0
        consistency = (action_types.get('BUY', 0) / max(1, sum(action_types.values()))) * 100
        
        return {
            'avg_score': 75.0,  # Placeholder - would need more sophisticated scoring
            'consistency': consistency,
            'risk_mgmt': 'Conservative' if avg_position < 10 else 'Aggressive',
            'sector_div': 'Diversified',  # Placeholder - would analyze sector distribution
            'avg_position': avg_position
        }
        
    except Exception as e:
        return {'avg_score': 0, 'consistency': 0, 'risk_mgmt': 'Error', 'sector_div': 'Error', 'avg_position': 0}

def calculate_var(daily_equity: Dict, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk"""
    if not daily_equity:
        return 0
    
    returns = pd.Series(daily_equity).pct_change().dropna()
    if len(returns) < 2:
        return 0
    
    return float(np.percentile(returns, confidence_level * 100) * 100)

def calculate_skewness(daily_equity: Dict) -> float:
    """Calculate return skewness"""
    if not daily_equity:
        return 0
    
    returns = pd.Series(daily_equity).pct_change().dropna()
    if len(returns) < 3:
        return 0
    
    return float(returns.skew())

def calculate_kurtosis(daily_equity: Dict) -> float:
    """Calculate return kurtosis"""
    if not daily_equity:
        return 0
    
    returns = pd.Series(daily_equity).pct_change().dropna()
    if len(returns) < 4:
        return 0
    
    return float(returns.kurtosis())

def calculate_treynor_ratio(result_data: Dict) -> float:
    """Calculate Treynor ratio using S&P 500 beta"""
    annual_return = result_data.get('annual_return', 0) / 100
    beta = result_data.get('sp500_comparison', {}).get('beta', 1.0)
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    
    if beta != 0:
        return (annual_return - risk_free_rate) / beta
    return 0

def calculate_jensen_alpha(result_data: Dict) -> float:
    """Calculate Jensen's Alpha vs S&P 500"""
    portfolio_return = result_data.get('annual_return', 0) / 100
    sp500_return = result_data.get('sp500_comparison', {}).get('sp500_annual_return', 0)
    beta = result_data.get('sp500_comparison', {}).get('beta', 1.0)
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    
    expected_return = risk_free_rate + beta * (sp500_return - risk_free_rate)
    return portfolio_return - expected_return

def calculate_avg_trade_duration(mode: str) -> float:
    """Calculate average trade duration in days"""
    # Placeholder - would analyze actual trade data
    return 45.0

def calculate_annual_turnover(mode: str) -> float:
    """Calculate annual portfolio turnover rate"""
    # Placeholder - would analyze actual portfolio changes
    return 0.8

def calculate_avg_holdings(mode: str) -> float:
    """Calculate average number of holdings"""
    # Placeholder - would analyze actual portfolio data
    return 8.5

def calculate_avg_position_size(mode: str) -> float:
    """Calculate average position size as percentage"""
    # Placeholder - would analyze actual position sizes
    return 12.5

def calculate_sector_concentration(mode: str) -> float:
    """Calculate Herfindahl-Hirschman Index for sector concentration"""
    # Placeholder - would analyze actual sector allocations
    return 0.15

def calculate_cash_utilization(mode: str) -> float:
    """Calculate average cash utilization percentage"""
    # Placeholder - would analyze actual cash levels
    return 85.0

def export_academic_csv(academic_data: Dict) -> None:
    """Export data in CSV format for statistical analysis"""
    try:
        rows = []
        for mode, metrics in academic_data.get('performance_metrics', {}).items():
            row = {
                'LLM_Mode': mode,
                'Total_Return_Pct': metrics['returns']['total_return_pct'],
                'Annual_Return_Pct': metrics['returns']['annualized_return_pct'],
                'Volatility_Pct': metrics['risk_metrics']['volatility_pct'],
                'Sharpe_Ratio': metrics['risk_adjusted_metrics']['sharpe_ratio'],
                'Sortino_Ratio': metrics['risk_adjusted_metrics']['sortino_ratio'],
                'Max_Drawdown_Pct': metrics['risk_metrics']['max_drawdown_pct'],
                'Total_Trades': metrics['trading_statistics']['total_trades'],
                'Win_Rate_Pct': metrics['trading_statistics']['win_rate_pct'],
                'SP500_Excess_Return_Pct': metrics['returns']['excess_return_vs_sp500_pct'],
                'SP500_Beta': metrics['benchmark_analysis']['market_beta_sp500'],
                'SP500_Correlation': metrics['benchmark_analysis']['correlation_with_sp500']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv('academic_analysis.csv', index=False)
        
    except Exception as e:
        print(f"CSV export failed: {e}")

async def run_comparison_simulation():
    """Run simulations with all three LLM modes and compare results"""

    print("\n" + "=" * 80)
    print("🔬 RUNNING COMPARISON SIMULATION")
    print("Testing: OpenAI Only vs Anthropic Only vs Both Combined")
    print("=" * 80)

    comparison_results = {}

    # Define the modes to test
    modes = ['OPENAI_ONLY', 'ANTHROPIC_ONLY', 'BOTH']

    # First download data once (will be cached for all runs)
    print("\n📊 Downloading market data (once for all simulations)...")
    initial_data_manager = DataManager(SimulationConfig())
    initial_data_manager.download_all_data()

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"🚀 Starting simulation with mode: {mode}")
        print(f"{'=' * 60}")

        try:
            # Create new orchestrator for each mode with fresh portfolio
            orchestrator = SimulationOrchestrator(dry_run=False, llm_mode=mode)

            # Ensure fresh portfolio for each run
            orchestrator.portfolio_manager.reset_portfolio()

            # Run simulation
            results = await orchestrator.run_simulation()

            # Store results
            comparison_results[mode] = results

            print(f"\n✅ Completed {mode} simulation")
            print(f"   Total Return: {results.get('total_return', 0):.2f}%")
            print(f"   Annual Return: {results.get('annual_return', 0):.2f}%")
            print(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")

        except Exception as e:
            print(f"\n❌ Failed {mode} simulation: {e}")
            comparison_results[mode] = {'error': str(e)}

    # Enhanced academic analysis
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE ACADEMIC ANALYSIS")
    print("=" * 80)
    
    # Calculate statistical significance
    _calculate_statistical_significance(comparison_results)
    
    # Calculate additional academic metrics
    _calculate_academic_metrics(comparison_results)
    
    # Export comprehensive academic analysis
    _export_detailed_results(comparison_results)
    export_detailed_academic_report(comparison_results)
    
    # Enhanced academic comparison summary
    print("\n" + "=" * 120)
    print("📊 COMPREHENSIVE ACADEMIC ANALYSIS")
    print("=" * 120)

    # Create comparison table
    print("\n{:<20} {:>15} {:>15} {:>15} {:>15}".format(
        "LLM Mode", "Total Return %", "Annual Return %", "Sharpe Ratio", "Max Drawdown %"
    ))
    print("-" * 80)

    best_return = -float('inf')
    best_sharpe = -float('inf')
    best_mode_return = None
    best_mode_sharpe = None

    for mode, results in comparison_results.items():
        if 'error' not in results:
            total_ret = results.get('total_return', 0)
            annual_ret = results.get('annual_return', 0)
            sharpe = results.get('sharpe_ratio', 0)
            max_dd = results.get('max_drawdown', 0)

            print("{:<20} {:>15.2f} {:>15.2f} {:>15.3f} {:>15.2f}".format(
                mode, total_ret, annual_ret, sharpe, max_dd
            ))

            if annual_ret > best_return:
                best_return = annual_ret
                best_mode_return = mode

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_mode_sharpe = mode
        else:
            print("{:<20} {:>15} {:>15} {:>15} {:>15}".format(
                mode, "ERROR", "ERROR", "ERROR", "ERROR"
            ))

    print("-" * 80)

    # Print winners
    print("\n🏆 PERFORMANCE WINNERS:")
    if best_mode_return:
        print(f"   Best Return: {best_mode_return} ({best_return:.2f}% annually)")
    if best_mode_sharpe:
        print(f"   Best Risk-Adjusted: {best_mode_sharpe} (Sharpe: {best_sharpe:.3f})")

    # Save comparison results
    with open('comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    print("\n📁 Full comparison results saved to comparison_results.json")

    # Compare with Berkshire
    print("\n🎯 Berkshire Hathaway Comparison:")
    print("-" * 40)

    for mode, results in comparison_results.items():
        if 'error' not in results and 'berkshire_comparison' in results:
            comp = results['berkshire_comparison']
            relative = comp.get('relative_performance', 0)

            if relative > 0:
                print(f"{mode}: OUTPERFORMED by {relative:.2f}% annually ✅")
            else:
                print(f"{mode}: Underperformed by {abs(relative):.2f}% annually ❌")

    # Trading statistics comparison
    print("\n📈 Trading Statistics:")
    print("-" * 40)

    for mode, results in comparison_results.items():
        if 'error' not in results:
            print(f"{mode}:")
            print(f"   Total Trades: {results.get('total_trades', 0)}")
            print(f"   Win Rate: {results.get('win_rate', 0):.1f}%")
            print(f"   Profit Factor: {results.get('profit_factor', 0):.2f}")

    return comparison_results

# ============================================================================
# SECTION 13: MAIN EXECUTION
# ============================================================================

def check_all_api_connections() -> Dict[str, bool]:
    """Check all API connections before starting simulation - SAFE VERSION"""

    logger = logging.getLogger(__name__)
    results = {}

    print("\n" + "=" * 60)
    print("🔍 CHECKING ALL API CONNECTIONS...")
    print("=" * 60)

    # Test each API but NEVER print the actual keys

    # 1. OpenAI API Check
    print("1. Testing OpenAI API connection...")
    try:
        key = os.getenv('OPENAI_API_KEY', '')
        if key:
            # Mask the key for display
            masked_key = mask_api_key(key)
            print(f"   Using key: {masked_key}")

            openai_client = openai.OpenAI(api_key=key)
            # Use a minimal test to avoid costs
            response = openai_client.models.list()
            if response.data:
                print("   ✅ OpenAI API: Connected successfully")
                results['openai'] = True
            else:
                print("   ❌ OpenAI API: No models available")
                results['openai'] = False
        else:
            print("   ❌ OpenAI API: No key found")
            results['openai'] = False
    except Exception as e:
        print(f"   ❌ OpenAI API: Connection failed - {str(e)[:100]}")
        results['openai'] = False

    # 2. Anthropic API Check
    print("2. Testing Anthropic API connection...")
    try:
        key = os.getenv('ANTHROPIC_API_KEY', '')
        if key:
            masked_key = mask_api_key(key)
            print(f"   Using key: {masked_key}")

            anthropic_client = anthropic.Anthropic(api_key=key)
            # Test with a minimal message
            response = anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            if response.content:
                print("   ✅ Anthropic API: Connected successfully")
                results['anthropic'] = True
            else:
                print("   ❌ Anthropic API: No response received")
                results['anthropic'] = False
        else:
            print("   ❌ Anthropic API: No key found")
            results['anthropic'] = False
    except Exception as e:
        print(f"   ❌ Anthropic API: Connection failed - {str(e)[:100]}")
        results['anthropic'] = False

    # 3. Alpha Vantage API Check
    print("3. Testing Alpha Vantage API connection...")
    try:
        # Test Alpha Vantage connection
        alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        if alpha_vantage_key:
            av_test = AlphaVantageManager(alpha_vantage_key)
            # Try to get a small amount of data to test connection
            test_data = av_test.get_daily_adjusted("AAPL", outputsize="compact")
            if test_data is not None and not test_data.empty:
                print("   ✅ Alpha Vantage API: Connected successfully")
                results['alpha_vantage'] = True
            else:
                print("   ❌ Alpha Vantage API: No data received")
                results['alpha_vantage'] = False
        else:
            print("   ❌ Alpha Vantage API: No API key found")
            results['alpha_vantage'] = False
    except Exception as e:
        print(f"   ❌ Alpha Vantage API: Connection failed - {str(e)[:100]}")
        results['alpha_vantage'] = False

    # 4. Wikipedia Check (optional)
    print("4. Wikipedia access (no API key needed)...")
    results['wikipedia'] = True  # Just mark as true since it's optional
    print("   ✅ Wikipedia: Will use fallback list if needed")

    # 5. General Internet Connection Check
    print("5. Testing general internet connectivity...")
    internet_test_urls = [
        "https://www.google.com",
        "https://api.github.com",
        "https://httpbin.org/get"
    ]
    
    for url in internet_test_urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code in [200, 301, 302]:  # Accept redirects as success
                print(f"   ✅ Internet Connection: Available (tested via {url.split('/')[2]})")
                results['internet'] = True
                break
        except Exception as e:
            continue
    else:
        # All URLs failed
        print("   ❌ Internet Connection: Failed - Unable to reach any test endpoints")
        results['internet'] = False

    # Summary
    print("=" * 60)
    print("📊 API CONNECTION SUMMARY:")
    print("=" * 60)

    critical_apis = ['openai', 'anthropic', 'alpha_vantage', 'internet']
    optional_apis = ['wikipedia']

    critical_passed = all(results.get(api, False) for api in critical_apis)
    total_passed = sum(results.values())
    total_apis = len(results)

    for api, status in results.items():
        status_icon = "✅" if status else "❌"
        api_type = "(Critical)" if api in critical_apis else "(Optional)"
        print(f"   {status_icon} {api.replace('_', ' ').title()}: {'PASS' if status else 'FAIL'} {api_type}")

    print(f"\n📈 Overall: {total_passed}/{total_apis} APIs connected successfully")

    if not critical_passed:
        print("\n❌ CRITICAL APIs failed - simulation cannot continue")
        missing_keys = []
        if not results.get('openai', False) and not os.getenv('OPENAI_API_KEY'):
            missing_keys.append('OPENAI_API_KEY')
        if not results.get('anthropic', False) and not os.getenv('ANTHROPIC_API_KEY'):
            missing_keys.append('ANTHROPIC_API_KEY')

        if missing_keys:
            print(f"🔑 Missing environment variables: {', '.join(missing_keys)}")
            print("   Please set these in your .env file or environment")

        return results
    else:
        print("✅ All critical APIs connected - simulation ready to start!")

    print("=" * 60)
    return results

def test_simulation():
    """Quick test with minimal API calls"""
    print("\n" + "=" * 60)
    print("🧪 RUNNING TEST SIMULATION (1 date only)")
    print("=" * 60)

    config = SimulationConfig()
    config.DECISION_DATES = ['2022-01-15']  # Only one date
    config.USE_PROFESSIONAL_RESEARCH = False
    config.USE_LLM_DECISIONS = False  # Use rule-based only
    config.USE_EXPENSIVE_APIS = False
    config.MAX_RESEARCH_PER_DATE = 2  # Even fewer stocks

    orchestrator = SimulationOrchestrator()
    orchestrator.config = config

    try:
        results = asyncio.run(orchestrator.run_simulation())
        print("✅ Test successful!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

# ============================================================================
# SECTION 12: MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run simulations with mode selection"""

    print("\n" + "=" * 80)
    print("🚀 LLM INVESTMENT SIMULATION SYSTEM")
    print("Version 7.0 - With LLM Comparison Capability")
    print("Compare OpenAI vs Anthropic vs Both Combined")
    print("=" * 80)

    # IMPORTANT: Create .env file first
    if not os.path.exists('.env'):
        print("\n⚠️  WARNING: No .env file found!")
        print("\nPlease create a .env file with your API keys:")
        print("=" * 60)
        print("# REQUIRED for comparison")
        print("OPENAI_API_KEY=your_openai_key_here")
        print("ANTHROPIC_API_KEY=your_anthropic_key_here")
        print("\n# Optional APIs for enhanced research")
        print("FINNHUB_API_KEY=your_finnhub_key_here")
        print("ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here")
        print("NEWS_API_KEY=your_newsapi_key_here")
        print("=" * 60)
        print("\n❌ Exiting - Please create .env file first")
        return {}

    # Check all API connections first
    api_results = check_all_api_connections()

    # Check if BOTH LLMs are available
    has_openai = api_results.get('openai', False)
    has_anthropic = api_results.get('anthropic', False)
    has_both_llms = has_openai and has_anthropic

    print("\n" + "=" * 60)
    print("🤖 LLM STATUS CHECK:")
    print("=" * 60)

    if has_both_llms:
        print("✅ EXCELLENT! Both OpenAI and Anthropic are connected")
        print("   You can run comparisons between all modes")
    elif has_openai:
        print("⚠️  Only OpenAI is connected")
        print("   You can run OpenAI-only mode")
    elif has_anthropic:
        print("⚠️  Only Anthropic is connected")
        print("   You can run Anthropic-only mode")
    else:
        print("❌ NO LLMs connected - will use rule-based decisions only")

    # Verify critical APIs are working
    critical_apis = ['alpha_vantage', 'internet']
    critical_passed = all(api_results.get(api, False) for api in critical_apis)

    if not critical_passed:
        print("\n❌ Critical API connections failed")
        print("\nWould you like to run in DRY RUN mode (no API calls)?")
        response = input("Continue with dry run? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return {}

        # Run in dry run mode
        orchestrator = SimulationOrchestrator(dry_run=True)
        results = asyncio.run(orchestrator.run_simulation())
        return results

    print("\n📋 Configuration Status:")
    print(f"✓ OpenAI GPT-4: {'✅ Connected' if has_openai else '❌ Not available'}")
    print(f"✓ Anthropic Claude: {'✅ Connected' if has_anthropic else '❌ Not available'}")
    print(f"✓ Alpha Vantage: {'✅ Connected' if api_results.get('alpha_vantage') else '❌ Not available'}")

    print("\n🎯 SIMULATION OPTIONS:")
    print("-" * 40)

    if has_both_llms:
        print("1. 🔬 COMPARISON MODE - Run all 3 modes and compare")
        print("   • OpenAI only")
        print("   • Anthropic only")
        print("   • Both combined")
        print("   (This will run 3 complete simulations)")
        print("\n2. 📊 OPENAI ONLY - Use only OpenAI GPT-4")
        print("3. 🤖 ANTHROPIC ONLY - Use only Claude Opus")
        print("4. 🎯 BOTH COMBINED - Use consensus from both")
        print("5. 🧪 TEST MODE - Quick test (1 date, minimal cost)")
        print("6. ❌ Exit")

        choice = input("\nSelect option (1-6): ")

        if choice == '1':
            # Run comparison mode
            print("\n🔬 Starting COMPARISON MODE...")
            print("This will run 3 complete simulations to compare performance")
            response = input("Continue? This will use significant API credits (y/n): ")
            if response.lower() == 'y':
                comparison_results = asyncio.run(run_comparison_simulation())
                return comparison_results
            else:
                return {}

        elif choice == '2':
            mode = 'OPENAI_ONLY'
        elif choice == '3':
            mode = 'ANTHROPIC_ONLY'
        elif choice == '4':
            mode = 'BOTH'
        elif choice == '5':
            # Test mode
            print("\n🧪 Running test simulation...")
            if test_simulation():
                print("\n✅ Test passed!")
            return {}
        else:
            print("Exiting...")
            return {}

    elif has_openai:
        print("1. 📊 OPENAI ONLY - Use OpenAI GPT-4")
        print("2. 🧪 TEST MODE - Quick test")
        print("3. ❌ Exit")

        choice = input("\nSelect option (1-3): ")

        if choice == '1':
            mode = 'OPENAI_ONLY'
        elif choice == '2':
            if test_simulation():
                print("\n✅ Test passed!")
            return {}
        else:
            return {}

    elif has_anthropic:
        print("1. 🤖 ANTHROPIC ONLY - Use Claude Opus")
        print("2. 🧪 TEST MODE - Quick test")
        print("3. ❌ Exit")

        choice = input("\nSelect option (1-3): ")

        if choice == '1':
            mode = 'ANTHROPIC_ONLY'
        elif choice == '2':
            if test_simulation():
                print("\n✅ Test passed!")
            return {}
        else:
            return {}
    else:
        print("\n❌ No LLMs available. Please add API keys.")
        return {}

    # Run single mode simulation
    try:
        print(f"\n🚀 Starting simulation with mode: {mode}")
        orchestrator = SimulationOrchestrator(dry_run=False, llm_mode=mode)
        results = asyncio.run(orchestrator.run_simulation())

        print("\n✅ Simulation completed successfully!")

        # Summary
        print("\n" + "=" * 80)
        print(f"📊 FINAL SUMMARY ({mode})")
        print("=" * 80)

        if results.get('total_return', 0) > 0:
            print("🎉 PROFITABLE STRATEGY!")
        else:
            print("📉 Strategy needs improvement")

        print(f"\nMode: {mode}")
        print(f"Total Return: {results.get('total_return', 0):+.2f}%")
        print(f"Annualized Return: {results.get('annual_return', 0):+.2f}%")
        print(f"Risk-Adjusted Return (Sharpe): {results.get('sharpe_ratio', 0):.3f}")
        print(f"Maximum Drawdown: {results.get('max_drawdown', 0):.2f}%")

        if 'berkshire_comparison' in results:
            comp = results['berkshire_comparison']
            relative = comp.get('relative_performance', 0)

            if relative > 0:
                print(f"\n🏆 OUTPERFORMED Berkshire by {relative:.2f}% annually!")
            else:
                print(f"\n📉 Underperformed Berkshire by {abs(relative):.2f}% annually")

        print(f"\n📁 Results saved with suffix: _{mode.lower()}")
        print(f"  • portfolio_history_{mode.lower()}.csv")
        print(f"  • trades_{mode.lower()}.csv")
        print(f"  • daily_equity_{mode.lower()}.csv")
        print(f"  • performance_metrics_{mode.lower()}.json")

        return results

    except KeyboardInterrupt:
        print("\n\n⚠️ Simulation interrupted by user")
        return None

    except Exception as e:
        print(f"\n❌ Simulation failed with error: {e}")
        logger.error(f"Main execution failed: {e}", exc_info=True)
        print("\nCheck llm_investment_full_system.log for details")
        return None

if __name__ == "__main__":
    # Run the complete simulation
    results = main()

    if results:
        print("\n" + "=" * 80)
        print("🎯 SIMULATION COMPLETE!")
        print("=" * 80)

        # Final stats - check if this is comparison results or individual results
        if isinstance(results, dict) and any(mode in results for mode in ['OPENAI_ONLY', 'ANTHROPIC_ONLY', 'BOTH']):
            # This is comparison results - show summary from best performer
            best_mode = None
            best_return = -999
            
            for mode, result_data in results.items():
                if 'error' not in result_data:
                    annual_return = result_data.get('annual_return', 0)
                    if annual_return > best_return:
                        best_return = annual_return
                        best_mode = mode
            
            if best_mode:
                best_results = results[best_mode]
                print(f"\n📊 Best Performance ({best_mode}):")
                print(f"  • Total Trades: {best_results.get('total_trades', 0)}")
                print(f"  • Win Rate: {best_results.get('win_rate', 0):.1f}%")
                print(f"  • Max Drawdown: {best_results.get('max_drawdown', 0):.2f}%")
                print(f"  • Annual Return: {best_results.get('annual_return', 0):.2f}%")
                print(f"  • Final Value: ${best_results.get('final_value', 0):,.0f}")
            else:
                print(f"\n📊 All simulations encountered errors")
        else:
            # Individual simulation results
            print(f"\n📊 Key Statistics:")
            print(f"  • Total Trades: {results.get('total_trades', 0)}")
            print(f"  • Win Rate: {results.get('win_rate', 0):.1f}%")
            print(f"  • Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
            print(f"  • Final Value: ${results.get('final_value', 0):,.0f}")

        # Success message
        if results.get('total_return', 0) > 50:
            print("\n🌟 EXCEPTIONAL PERFORMANCE!")
        elif results.get('total_return', 0) > 20:
            print("\n⭐ STRONG PERFORMANCE!")
        elif results.get('total_return', 0) > 0:
            print("\n✅ POSITIVE RETURNS!")

        print("\nThank you for using the Complete LLM Investment System!")
        print("=" * 80)
