"""
Volatility calculation utilities for options trading strategies.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional


def historical_volatility(
    prices: Union[pd.Series, np.ndarray],
    window: int = 30,
    annualization_factor: int = 252
) -> Union[pd.Series, float]:
    """
    Calculate historical volatility using log returns.

    Args:
        prices: Price series or array
        window: Rolling window size (default: 30 days)
        annualization_factor: Days per year for annualization (default: 252)

    Returns:
        Historical volatility (annualized)
    """
    if isinstance(prices, pd.Series):
        log_returns = np.log(prices / prices.shift(1))
        hv = log_returns.rolling(window=window).std() * np.sqrt(annualization_factor)
        return hv
    else:
        log_returns = np.log(prices[1:] / prices[:-1])
        if len(log_returns) < window:
            return np.nan
        hv = np.std(log_returns[-window:]) * np.sqrt(annualization_factor)
        return hv


def realized_volatility(
    prices: Union[pd.Series, np.ndarray],
    annualization_factor: int = 252
) -> float:
    """
    Calculate realized volatility over the entire period.

    Args:
        prices: Price series or array
        annualization_factor: Days per year for annualization (default: 252)

    Returns:
        Realized volatility (annualized)
    """
    if isinstance(prices, pd.Series):
        log_returns = np.log(prices / prices.shift(1)).dropna()
    else:
        log_returns = np.log(prices[1:] / prices[:-1])

    return np.std(log_returns) * np.sqrt(annualization_factor)


def parkinson_volatility(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    window: int = 30,
    annualization_factor: int = 252
) -> Union[pd.Series, float]:
    """
    Calculate Parkinson's volatility (uses high-low range).
    More efficient than close-to-close volatility.

    Args:
        high: High prices
        low: Low prices
        window: Rolling window size
        annualization_factor: Days per year for annualization

    Returns:
        Parkinson volatility (annualized)
    """
    hl_ratio = np.log(high / low)

    if isinstance(high, pd.Series):
        pv = (hl_ratio ** 2).rolling(window=window).mean()
        pv = np.sqrt(pv / (4 * np.log(2))) * np.sqrt(annualization_factor)
        return pv
    else:
        if len(hl_ratio) < window:
            return np.nan
        pv = np.mean(hl_ratio[-window:] ** 2)
        pv = np.sqrt(pv / (4 * np.log(2))) * np.sqrt(annualization_factor)
        return pv


def garman_klass_volatility(
    open_price: Union[pd.Series, np.ndarray],
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    window: int = 30,
    annualization_factor: int = 252
) -> Union[pd.Series, float]:
    """
    Calculate Garman-Klass volatility (uses OHLC data).
    More efficient estimator than Parkinson.

    Args:
        open_price: Opening prices
        high: High prices
        low: Low prices
        close: Closing prices
        window: Rolling window size
        annualization_factor: Days per year for annualization

    Returns:
        Garman-Klass volatility (annualized)
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_price)

    gk = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)

    if isinstance(high, pd.Series):
        gkv = gk.rolling(window=window).mean()
        gkv = np.sqrt(gkv) * np.sqrt(annualization_factor)
        return gkv
    else:
        if len(gk) < window:
            return np.nan
        gkv = np.mean(gk[-window:])
        gkv = np.sqrt(gkv) * np.sqrt(annualization_factor)
        return gkv


def yang_zhang_volatility(
    open_price: Union[pd.Series, np.ndarray],
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    window: int = 30,
    annualization_factor: int = 252
) -> Union[pd.Series, float]:
    """
    Calculate Yang-Zhang volatility (handles overnight jumps).
    One of the most accurate volatility estimators.

    Args:
        open_price: Opening prices
        high: High prices
        low: Low prices
        close: Closing prices
        window: Rolling window size
        annualization_factor: Days per year for annualization

    Returns:
        Yang-Zhang volatility (annualized)
    """
    log_ho = np.log(high / open_price)
    log_lo = np.log(low / open_price)
    log_co = np.log(close / open_price)

    if isinstance(open_price, pd.Series):
        log_oc = np.log(open_price / close.shift(1))
        log_cc = np.log(close / close.shift(1))

        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        open_var = (log_oc ** 2).rolling(window=window).mean()
        close_var = (log_cc ** 2).rolling(window=window).mean()
        rs_mean = rs.rolling(window=window).mean()

        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_var = open_var + k * close_var + (1 - k) * rs_mean

        return np.sqrt(yz_var) * np.sqrt(annualization_factor)
    else:
        if len(open_price) < window + 1:
            return np.nan

        log_oc = np.log(open_price[1:] / close[:-1])
        log_cc = np.log(close[1:] / close[:-1])

        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        open_var = np.var(log_oc[-(window):])
        close_var = np.var(log_cc[-(window):])
        rs_mean = np.mean(rs[-(window):])

        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_var = open_var + k * close_var + (1 - k) * rs_mean

        return np.sqrt(yz_var) * np.sqrt(annualization_factor)


def ewma_volatility(
    prices: Union[pd.Series, np.ndarray],
    span: int = 30,
    annualization_factor: int = 252
) -> Union[pd.Series, float]:
    """
    Calculate EWMA (Exponentially Weighted Moving Average) volatility.
    Gives more weight to recent observations.

    Args:
        prices: Price series or array
        span: Span for EWMA calculation
        annualization_factor: Days per year for annualization

    Returns:
        EWMA volatility (annualized)
    """
    if isinstance(prices, pd.Series):
        log_returns = np.log(prices / prices.shift(1)).dropna()
        ewma_var = log_returns.ewm(span=span).var()
        return np.sqrt(ewma_var) * np.sqrt(annualization_factor)
    else:
        log_returns = np.log(prices[1:] / prices[:-1])
        alpha = 2 / (span + 1)

        ewma_var = log_returns[0] ** 2
        for ret in log_returns[1:]:
            ewma_var = alpha * (ret ** 2) + (1 - alpha) * ewma_var

        return np.sqrt(ewma_var) * np.sqrt(annualization_factor)


def volatility_cone(
    prices: pd.Series,
    windows: list = [10, 20, 30, 60, 90, 120],
    percentiles: list = [10, 25, 50, 75, 90]
) -> pd.DataFrame:
    """
    Calculate volatility cone showing historical volatility ranges.

    Args:
        prices: Price series
        windows: List of window sizes to analyze
        percentiles: Percentiles to calculate

    Returns:
        DataFrame with volatility statistics for each window
    """
    results = []

    for window in windows:
        hv = historical_volatility(prices, window=window)
        hv_valid = hv.dropna()

        if len(hv_valid) > 0:
            stats = {
                'window': window,
                'current': hv_valid.iloc[-1],
                'min': hv_valid.min(),
                'max': hv_valid.max(),
                'mean': hv_valid.mean(),
            }

            for p in percentiles:
                stats[f'p{p}'] = np.percentile(hv_valid, p)

            results.append(stats)

    return pd.DataFrame(results)
