"""
Market data fetching and processing utilities.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from ..utils.volatility import historical_volatility
from ..utils.black_scholes import implied_volatility


class MarketDataFetcher:
    """
    Fetches and processes market data for volatility arbitrage strategies.
    """

    def __init__(self):
        """Initialize market data fetcher."""
        self.cache = {}

    def get_stock_data(
        self,
        ticker: str,
        period: str = '1y',
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch historical stock data.

        Args:
            ticker: Stock ticker symbol
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d', '1wk', '1mo')

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{ticker}_{period}_{interval}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for {ticker}")

        self.cache[cache_key] = df
        return df

    def get_current_price(self, ticker: str) -> float:
        """
        Get current stock price.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current price
        """
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m')

        if data.empty:
            raise ValueError(f"No data found for {ticker}")

        return data['Close'].iloc[-1]

    def get_options_chain(
        self,
        ticker: str,
        expiration_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch options chain for a stock.

        Args:
            ticker: Stock ticker symbol
            expiration_date: Specific expiration date (YYYY-MM-DD format)
                           If None, uses nearest expiration

        Returns:
            Tuple of (calls_df, puts_df)
        """
        stock = yf.Ticker(ticker)
        expirations = stock.options

        if len(expirations) == 0:
            raise ValueError(f"No options data available for {ticker}")

        if expiration_date is None:
            expiration_date = expirations[0]
        elif expiration_date not in expirations:
            raise ValueError(f"Invalid expiration date. Available: {expirations}")

        opt = stock.option_chain(expiration_date)
        return opt.calls, opt.puts

    def get_available_expirations(self, ticker: str) -> List[str]:
        """
        Get list of available option expiration dates.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of expiration dates
        """
        stock = yf.Ticker(ticker)
        return list(stock.options)

    def calculate_historical_volatility_series(
        self,
        ticker: str,
        period: str = '1y',
        windows: List[int] = [10, 20, 30, 60]
    ) -> pd.DataFrame:
        """
        Calculate historical volatility for multiple windows.

        Args:
            ticker: Stock ticker symbol
            period: Data period
            windows: List of window sizes (in days)

        Returns:
            DataFrame with historical volatility for each window
        """
        df = self.get_stock_data(ticker, period=period)
        prices = df['Close']

        result = pd.DataFrame(index=df.index)

        for window in windows:
            hv = historical_volatility(prices, window=window)
            result[f'HV_{window}'] = hv

        return result

    def get_implied_volatility_from_options(
        self,
        ticker: str,
        expiration_date: Optional[str] = None,
        moneyness_range: Tuple[float, float] = (0.95, 1.05)
    ) -> Dict:
        """
        Calculate implied volatility from options chain.

        Args:
            ticker: Stock ticker symbol
            expiration_date: Expiration date
            moneyness_range: Range of moneyness (S/K) to consider for ATM options

        Returns:
            Dictionary with implied volatility metrics
        """
        # Get current price
        current_price = self.get_current_price(ticker)

        # Get options chain
        calls, puts = self.get_options_chain(ticker, expiration_date)

        if expiration_date is None:
            stock = yf.Ticker(ticker)
            expiration_date = stock.options[0]

        # Calculate time to expiration
        exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
        today = datetime.now()
        T = (exp_date - today).days / 365.0

        if T <= 0:
            raise ValueError("Expiration date is in the past")

        # Filter for near-the-money options
        calls['moneyness'] = current_price / calls['strike']
        puts['moneyness'] = current_price / puts['strike']

        atm_calls = calls[
            (calls['moneyness'] >= moneyness_range[0]) &
            (calls['moneyness'] <= moneyness_range[1])
        ]
        atm_puts = puts[
            (puts['moneyness'] >= moneyness_range[0]) &
            (puts['moneyness'] <= moneyness_range[1])
        ]

        # Use implied volatility from options if available
        ivs = []

        if 'impliedVolatility' in atm_calls.columns:
            ivs.extend(atm_calls['impliedVolatility'].dropna().tolist())
        if 'impliedVolatility' in atm_puts.columns:
            ivs.extend(atm_puts['impliedVolatility'].dropna().tolist())

        if len(ivs) == 0:
            # Calculate implied volatility from prices if not provided
            # This is a simplified approach - in practice you'd want more robust calculation
            return {
                'atm_iv': np.nan,
                'call_iv': np.nan,
                'put_iv': np.nan,
                'iv_skew': np.nan,
                'time_to_expiration': T,
                'current_price': current_price
            }

        call_ivs = atm_calls['impliedVolatility'].dropna().tolist() if 'impliedVolatility' in atm_calls.columns else []
        put_ivs = atm_puts['impliedVolatility'].dropna().tolist() if 'impliedVolatility' in atm_puts.columns else []

        return {
            'atm_iv': np.mean(ivs) if ivs else np.nan,
            'call_iv': np.mean(call_ivs) if call_ivs else np.nan,
            'put_iv': np.mean(put_ivs) if put_ivs else np.nan,
            'iv_skew': (np.mean(put_ivs) - np.mean(call_ivs)) if (call_ivs and put_ivs) else np.nan,
            'time_to_expiration': T,
            'current_price': current_price,
            'num_calls': len(atm_calls),
            'num_puts': len(atm_puts)
        }

    def get_volatility_surface(
        self,
        ticker: str,
        num_expirations: int = 5
    ) -> pd.DataFrame:
        """
        Build a volatility surface from options data.

        Args:
            ticker: Stock ticker symbol
            num_expirations: Number of expiration dates to include

        Returns:
            DataFrame with volatility surface
        """
        current_price = self.get_current_price(ticker)
        expirations = self.get_available_expirations(ticker)

        if len(expirations) == 0:
            raise ValueError(f"No options available for {ticker}")

        expirations = expirations[:num_expirations]

        surface_data = []

        for exp_date in expirations:
            calls, puts = self.get_options_chain(ticker, exp_date)

            # Calculate time to expiration
            exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
            T = (exp_dt - datetime.now()).days / 365.0

            if T <= 0:
                continue

            # Process calls
            for _, call in calls.iterrows():
                if 'impliedVolatility' in call and pd.notna(call['impliedVolatility']):
                    moneyness = current_price / call['strike']
                    surface_data.append({
                        'expiration': exp_date,
                        'T': T,
                        'strike': call['strike'],
                        'moneyness': moneyness,
                        'option_type': 'call',
                        'implied_vol': call['impliedVolatility'],
                        'volume': call.get('volume', 0),
                        'open_interest': call.get('openInterest', 0)
                    })

            # Process puts
            for _, put in puts.iterrows():
                if 'impliedVolatility' in put and pd.notna(put['impliedVolatility']):
                    moneyness = current_price / put['strike']
                    surface_data.append({
                        'expiration': exp_date,
                        'T': T,
                        'strike': put['strike'],
                        'moneyness': moneyness,
                        'option_type': 'put',
                        'implied_vol': put['impliedVolatility'],
                        'volume': put.get('volume', 0),
                        'open_interest': put.get('openInterest', 0)
                    })

        return pd.DataFrame(surface_data)

    def compare_implied_vs_historical(
        self,
        ticker: str,
        expiration_date: Optional[str] = None,
        hv_window: int = 30
    ) -> Dict:
        """
        Compare implied volatility vs historical volatility.

        Args:
            ticker: Stock ticker symbol
            expiration_date: Options expiration date
            hv_window: Window for historical volatility calculation

        Returns:
            Dictionary with comparison metrics
        """
        # Get historical volatility
        df = self.get_stock_data(ticker, period='1y')
        prices = df['Close']
        hv = historical_volatility(prices, window=hv_window).iloc[-1]

        # Get implied volatility
        iv_data = self.get_implied_volatility_from_options(ticker, expiration_date)
        iv = iv_data['atm_iv']

        if np.isnan(iv) or np.isnan(hv):
            return {
                'ticker': ticker,
                'implied_vol': iv,
                'historical_vol': hv,
                'vol_spread': np.nan,
                'vol_ratio': np.nan,
                'signal': 'NO_DATA'
            }

        vol_spread = iv - hv
        vol_ratio = iv / hv if hv > 0 else np.inf

        # Generate signal
        if vol_spread < -0.05:  # IV significantly below HV
            signal = 'BUY_VOLATILITY'
        elif vol_spread > 0.05:  # IV significantly above HV
            signal = 'SELL_VOLATILITY'
        else:
            signal = 'NEUTRAL'

        return {
            'ticker': ticker,
            'current_price': iv_data['current_price'],
            'implied_vol': iv,
            'historical_vol': hv,
            'vol_spread': vol_spread,
            'vol_ratio': vol_ratio,
            'signal': signal,
            'time_to_expiration': iv_data['time_to_expiration']
        }

    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate (simplified - uses a constant).

        In practice, you would fetch this from treasury rates.

        Returns:
            Risk-free rate (annualized)
        """
        # Simplified: using a constant rate
        # In production, fetch from ^TNX (10-year treasury) or ^IRX (13-week treasury)
        return 0.05  # 5%

    def clear_cache(self):
        """Clear the data cache."""
        self.cache = {}
