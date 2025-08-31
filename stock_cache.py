import os
import pickle
import json
import hashlib
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class StockDataCache:
    """Manages local caching of stock data to reduce API calls and improve performance"""
    
    def __init__(self, cache_dir: str = "stock_data_cache", cache_expiry_days: int = 7):
        """
        Initialize the stock data cache
        
        Args:
            cache_dir: Directory to store cached data
            cache_expiry_days: Number of days before cache expires
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_expiry = timedelta(days=cache_expiry_days)
        
        # Create subdirectories for different data types
        self.price_cache_dir = self.cache_dir / "prices"
        self.fundamental_cache_dir = self.cache_dir / "fundamentals"
        self.metadata_cache_dir = self.cache_dir / "metadata"
        
        for dir_path in [self.price_cache_dir, self.fundamental_cache_dir, self.metadata_cache_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata"""
        metadata_file = self.metadata_cache_dir / "cache_info.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        metadata_file = self.metadata_cache_dir / "cache_info.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, ticker: str, data_type: str, start_date: str = None, end_date: str = None) -> str:
        """Generate a unique cache key for the data"""
        key_parts = [ticker, data_type]
        if start_date:
            key_parts.append(start_date)
        if end_date:
            key_parts.append(end_date)
        
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.metadata:
            return False
        
        cache_time = datetime.fromisoformat(self.metadata[cache_key]['timestamp'])
        return datetime.now() - cache_time < self.cache_expiry
    
    def get_price_data(self, ticker: str, start_date: str, end_date: str, 
                       force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get price data from cache or download if needed
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            force_refresh: Force download even if cache exists
        
        Returns:
            DataFrame with price data or None if failed
        """
        cache_key = self._get_cache_key(ticker, "price", start_date, end_date)
        cache_file = self.price_cache_dir / f"{cache_key}.pkl"
        
        # Check if we have valid cached data
        if not force_refresh and cache_file.exists() and self._is_cache_valid(cache_key):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                logger.debug(f"Loaded {ticker} price data from cache")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached data for {ticker}: {e}")
        
        # Download fresh data
        try:
            logger.info(f"Downloading fresh price data for {ticker}")
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"No data available for {ticker}")
                return None
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(hist, f)
            
            # Update metadata
            self.metadata[cache_key] = {
                'ticker': ticker,
                'type': 'price',
                'start_date': start_date,
                'end_date': end_date,
                'timestamp': datetime.now().isoformat(),
                'records': len(hist)
            }
            self._save_metadata()
            
            return hist
            
        except Exception as e:
            logger.error(f"Failed to download price data for {ticker}: {e}")
            return None
    
    def get_fundamental_data(self, ticker: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get fundamental data from cache or download if needed
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: Force download even if cache exists
        
        Returns:
            Dictionary with fundamental data or None if failed
        """
        cache_key = self._get_cache_key(ticker, "fundamental")
        cache_file = self.fundamental_cache_dir / f"{cache_key}.json"
        
        # Check if we have valid cached data
        if not force_refresh and cache_file.exists() and self._is_cache_valid(cache_key):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                logger.debug(f"Loaded {ticker} fundamental data from cache")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached fundamental data for {ticker}: {e}")
        
        # Download fresh data
        try:
            logger.info(f"Downloading fresh fundamental data for {ticker}")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                logger.warning(f"No fundamental data available for {ticker}")
                return None
            
            # Extract relevant fundamental data
            fundamental_data = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'roe': info.get('returnOnEquity'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'gross_margins': info.get('grossMargins'),
                'operating_margins': info.get('operatingMargins'),
                'profit_margins': info.get('profitMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(fundamental_data, f, indent=2)
            
            # Update metadata
            self.metadata[cache_key] = {
                'ticker': ticker,
                'type': 'fundamental',
                'timestamp': datetime.now().isoformat()
            }
            self._save_metadata()
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Failed to download fundamental data for {ticker}: {e}")
            return None
    
    def get_all_stock_data(self, ticker: str, start_date: str, end_date: str, 
                          force_refresh: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Get both price and fundamental data for a stock
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            force_refresh: Force download even if cache exists
        
        Returns:
            Tuple of (price_data, fundamental_data)
        """
        price_data = self.get_price_data(ticker, start_date, end_date, force_refresh)
        fundamental_data = self.get_fundamental_data(ticker, force_refresh)
        
        return price_data, fundamental_data
    
    def filter_data_by_date(self, data: pd.DataFrame, decision_date: str) -> pd.DataFrame:
        """
        Filter data to only include dates up to and including the decision date
        
        Args:
            data: DataFrame with datetime index
            decision_date: Date string to filter up to
        
        Returns:
            Filtered DataFrame
        """
        decision_dt = pd.to_datetime(decision_date)
        return data[data.index <= decision_dt]
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cached data
        
        Args:
            older_than_days: Only clear cache older than this many days. If None, clear all.
        """
        cleared_count = 0
        
        if older_than_days is None:
            # Clear all cache
            for cache_dir in [self.price_cache_dir, self.fundamental_cache_dir]:
                for file in cache_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                        cleared_count += 1
            
            self.metadata = {}
            self._save_metadata()
            logger.info(f"Cleared all cache ({cleared_count} files)")
        else:
            # Clear old cache
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            keys_to_remove = []
            
            for cache_key, info in self.metadata.items():
                cache_time = datetime.fromisoformat(info['timestamp'])
                if cache_time < cutoff_date:
                    keys_to_remove.append(cache_key)
                    
                    # Delete the actual file
                    if info['type'] == 'price':
                        cache_file = self.price_cache_dir / f"{cache_key}.pkl"
                    else:
                        cache_file = self.fundamental_cache_dir / f"{cache_key}.json"
                    
                    if cache_file.exists():
                        cache_file.unlink()
                        cleared_count += 1
            
            for key in keys_to_remove:
                del self.metadata[key]
            
            self._save_metadata()
            logger.info(f"Cleared {cleared_count} old cache files")
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache"""
        total_files = 0
        total_size = 0
        
        for cache_dir in [self.price_cache_dir, self.fundamental_cache_dir]:
            for file in cache_dir.glob("*"):
                if file.is_file():
                    total_files += 1
                    total_size += file.stat().st_size
        
        return {
            'total_files': total_files,
            'total_size_mb': total_size / (1024 * 1024),
            'cached_tickers': len(set(info['ticker'] for info in self.metadata.values())),
            'oldest_cache': min((info['timestamp'] for info in self.metadata.values()), default=None),
            'newest_cache': max((info['timestamp'] for info in self.metadata.values()), default=None)
        }
