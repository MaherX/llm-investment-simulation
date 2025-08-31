"""
Data filtering utilities to ensure the LLM only sees historical data up to the decision date
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class HistoricalDataFilter:
    """Ensures all data passed to LLM is filtered to respect the decision date"""
    
    @staticmethod
    def filter_price_data(price_data: pd.DataFrame, decision_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Filter price data to only include dates up to and including the decision date
        
        Args:
            price_data: DataFrame with datetime index
            decision_date: The decision date to filter up to
            
        Returns:
            Filtered DataFrame
        """
        if isinstance(decision_date, str):
            decision_date = pd.to_datetime(decision_date)
        
        # Check if we have a DatetimeIndex before checking timezone
        if isinstance(price_data.index, pd.DatetimeIndex):
            # Handle timezone-aware vs timezone-naive
            if price_data.index.tz is not None and decision_date.tz is None:
                decision_date = decision_date.tz_localize('UTC')
            elif price_data.index.tz is None and decision_date.tz is not None:
                decision_date = decision_date.tz_localize(None)
        
        # Check if DataFrame is empty or doesn't have a proper DatetimeIndex
        if len(price_data) == 0 or not isinstance(price_data.index, pd.DatetimeIndex):
            # Return empty DataFrame for empty or improperly indexed data
            return price_data
        
        # Filter data
        filtered = price_data[price_data.index <= decision_date]
        
        # Only warn if we have insufficient data for meaningful analysis
        if len(filtered) == 0 and len(price_data) > 0:
            # Data exists but none before decision date - this might be an issue
            logger.debug(f"No data available before {decision_date} (data starts from {price_data.index[0]})")
        elif len(filtered) < 60:
            # Limited data - log at debug level since this is common for newer stocks
            logger.debug(f"Limited data available before {decision_date}: only {len(filtered)} days")
        
        return filtered
    
    @staticmethod
    def filter_fundamental_data(fundamental_data: Dict[str, Any], decision_date: Union[str, datetime]) -> Dict[str, Any]:
        """
        Filter fundamental data to remove any forward-looking information
        
        Args:
            fundamental_data: Dictionary of fundamental metrics
            decision_date: The decision date
            
        Returns:
            Filtered fundamental data
        """
        # List of metrics that should be considered historical
        # These are typically trailing metrics or static company info
        historical_metrics = [
            'market_cap', 'pe_ratio', 'pb_ratio', 'dividend_yield',
            'profit_margin', 'roe', 'debt_to_equity', 'revenue_growth',
            'earnings_growth', 'sector', 'industry', 'current_ratio',
            'gross_margins', 'operating_margins', 'profit_margins'
        ]
        
        # List of metrics that might contain forward-looking information
        # These should be excluded or marked as estimates
        forward_looking_metrics = [
            'targetMeanPrice', 'recommendationMean', 'numberOfAnalystOpinions',
            'forwardPE', 'forwardEps', 'pegRatio'
        ]
        
        filtered_data = {}
        
        for key, value in fundamental_data.items():
            # Only include historical metrics
            if key in historical_metrics:
                filtered_data[key] = value
            elif key in forward_looking_metrics:
                # Skip forward-looking metrics
                logger.debug(f"Excluding forward-looking metric: {key}")
                continue
            else:
                # For unknown metrics, include but log for review
                filtered_data[key] = value
                logger.debug(f"Including metric (review if historical): {key}")
        
        return filtered_data
    
    @staticmethod
    def filter_research_data(research_data: Dict[str, Any], decision_date: Union[str, datetime]) -> Dict[str, Any]:
        """
        Filter research data to ensure no future information is included
        
        Args:
            research_data: Dictionary containing research results
            decision_date: The decision date
            
        Returns:
            Filtered research data
        """
        if isinstance(decision_date, str):
            decision_date = pd.to_datetime(decision_date)
        
        filtered_research = research_data.copy()
        
        # Filter news items if present
        if 'news_sentiment' in filtered_research and 'articles' in filtered_research['news_sentiment']:
            filtered_articles = []
            for article in filtered_research['news_sentiment']['articles']:
                if 'date' in article:
                    article_date = pd.to_datetime(article['date'])
                    if article_date <= decision_date:
                        filtered_articles.append(article)
            filtered_research['news_sentiment']['articles'] = filtered_articles
        
        # Filter SEC filings if present
        if 'sec_filings' in filtered_research and 'recent_filings' in filtered_research['sec_filings']:
            filtered_filings = []
            for filing in filtered_research['sec_filings']['recent_filings']:
                if 'date' in filing:
                    filing_date = pd.to_datetime(filing['date'])
                    if filing_date <= decision_date:
                        filtered_filings.append(filing)
            filtered_research['sec_filings']['recent_filings'] = filtered_filings
        
        # Filter analyst opinions if present
        if 'analyst_opinions' in filtered_research and 'ratings' in filtered_research['analyst_opinions']:
            filtered_ratings = []
            for rating in filtered_research['analyst_opinions']['ratings']:
                if 'date' in rating:
                    rating_date = pd.to_datetime(rating['date'])
                    if rating_date <= decision_date:
                        filtered_ratings.append(rating)
            filtered_research['analyst_opinions']['ratings'] = filtered_ratings
        
        return filtered_research
    
    @staticmethod
    def prepare_historical_context(ticker: str, price_data: pd.DataFrame, 
                                  fundamental_data: Dict[str, Any],
                                  decision_date: Union[str, datetime]) -> Dict[str, Any]:
        """
        Prepare a complete historical context for a stock up to the decision date
        
        Args:
            ticker: Stock ticker
            price_data: Full price data DataFrame
            fundamental_data: Full fundamental data
            decision_date: The decision date
            
        Returns:
            Dictionary with properly filtered historical context
        """
        # Filter price data
        filtered_prices = HistoricalDataFilter.filter_price_data(price_data, decision_date)
        
        # Filter fundamental data
        filtered_fundamentals = HistoricalDataFilter.filter_fundamental_data(
            fundamental_data, decision_date
        )
        
        # Calculate historical metrics from filtered price data
        if len(filtered_prices) > 0:
            recent_price = filtered_prices['Close'].iloc[-1]
            
            # Calculate returns over different periods (all historical)
            returns = {}
            if len(filtered_prices) >= 252:  # 1 year
                returns['1y_return'] = (recent_price / filtered_prices['Close'].iloc[-252] - 1)
            if len(filtered_prices) >= 126:  # 6 months
                returns['6m_return'] = (recent_price / filtered_prices['Close'].iloc[-126] - 1)
            if len(filtered_prices) >= 63:  # 3 months
                returns['3m_return'] = (recent_price / filtered_prices['Close'].iloc[-63] - 1)
            if len(filtered_prices) >= 21:  # 1 month
                returns['1m_return'] = (recent_price / filtered_prices['Close'].iloc[-21] - 1)
            
            # Calculate volatility (historical)
            if len(filtered_prices) >= 60:
                returns_series = filtered_prices['Close'].pct_change().dropna()
                volatility = returns_series.tail(60).std() * (252 ** 0.5)  # Annualized
            else:
                volatility = None
            
            # Volume metrics (historical)
            if 'Volume' in filtered_prices.columns:
                avg_volume_20d = filtered_prices['Volume'].tail(20).mean()
                avg_volume_60d = filtered_prices['Volume'].tail(60).mean() if len(filtered_prices) >= 60 else None
            else:
                avg_volume_20d = None
                avg_volume_60d = None
        else:
            recent_price = None
            returns = {}
            volatility = None
            avg_volume_20d = None
            avg_volume_60d = None
        
        context = {
            'ticker': ticker,
            'decision_date': str(decision_date),
            'price_data_points': len(filtered_prices),
            'current_price': recent_price,
            'historical_returns': returns,
            'volatility_60d': volatility,
            'avg_volume_20d': avg_volume_20d,
            'avg_volume_60d': avg_volume_60d,
            'fundamentals': filtered_fundamentals,
            'data_cutoff_date': str(filtered_prices.index[-1]) if len(filtered_prices) > 0 else None
        }
        
        return context
    
    @staticmethod
    def validate_no_future_data(data: Dict[str, Any], decision_date: Union[str, datetime]) -> bool:
        """
        Validate that no future data is present in the dataset
        
        Args:
            data: Data dictionary to validate
            decision_date: The decision date
            
        Returns:
            True if no future data found, False otherwise
        """
        if isinstance(decision_date, str):
            decision_date = pd.to_datetime(decision_date)
        
        def check_dates(obj, path=""):
            """Recursively check for dates in the data structure"""
            issues = []
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if 'date' in key.lower() or 'time' in key.lower():
                        try:
                            date_val = pd.to_datetime(value)
                            if date_val > decision_date:
                                issues.append(f"Future date found at {path}.{key}: {date_val}")
                        except:
                            pass
                    issues.extend(check_dates(value, f"{path}.{key}"))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    issues.extend(check_dates(item, f"{path}[{i}]"))
            elif isinstance(obj, pd.DataFrame):
                if len(obj) > 0 and obj.index.name and 'date' in str(obj.index.name).lower():
                    future_dates = obj.index[obj.index > decision_date]
                    if len(future_dates) > 0:
                        issues.append(f"Future dates in DataFrame at {path}: {future_dates[:5].tolist()}")
            
            return issues
        
        issues = check_dates(data)
        
        if issues:
            logger.warning(f"Found {len(issues)} future data issues:")
            for issue in issues[:5]:  # Show first 5 issues
                logger.warning(f"  - {issue}")
            return False
        
        return True
