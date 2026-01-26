"""
Cross-Sectional Momentum Signal.

This module implements the classic cross-sectional momentum strategy:
- Calculate momentum as cumulative returns over lookback period (skipping recent month)
- Rank stocks cross-sectionally
- Go long top decile, short bottom decile
- Maintain dollar neutrality

Enhancements:
- Volatility scaling per stock
- Winsorize extreme returns
- Industry/sector neutralization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class MomentumType(Enum):
    """Types of momentum calculation."""
    CUMULATIVE = "cumulative"  # Sum of log returns
    TOTAL_RETURN = "total_return"  # (P_t - P_0) / P_0
    RISK_ADJUSTED = "risk_adjusted"  # Sharpe-like momentum


@dataclass
class MomentumConfig:
    """Configuration for momentum signal calculation."""
    # Lookback parameters
    lookback_start: int = 252  # Start of lookback window (days ago)
    lookback_end: int = 21     # End of lookback window (skip recent month)

    # Momentum type
    momentum_type: MomentumType = MomentumType.CUMULATIVE

    # Portfolio construction
    long_percentile: float = 0.1   # Top 10% (decile)
    short_percentile: float = 0.1  # Bottom 10%

    # Enhancements
    volatility_scale: bool = True  # Scale signals by inverse volatility
    vol_lookback: int = 60         # Lookback for volatility estimation
    winsorize: bool = True         # Winsorize extreme returns
    winsorize_pct: float = 0.01    # Winsorize at 1% and 99%
    sector_neutralize: bool = False  # Industry/sector neutralization

    # Risk management
    max_position_weight: float = 0.05  # Max weight per position (5%)
    min_stocks_per_side: int = 5       # Minimum stocks in long/short

    # Annualization
    annualization_factor: int = 252


@dataclass
class PortfolioWeights:
    """Container for portfolio weights."""
    long_weights: Dict[str, float] = field(default_factory=dict)
    short_weights: Dict[str, float] = field(default_factory=dict)
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    long_count: int = 0
    short_count: int = 0


class CrossSectionalMomentum:
    """
    Cross-Sectional Momentum Signal Generator.

    The classic momentum strategy:
    1. For each stock i at time t, calculate:
       mom_{i,t} = sum_{k=skip+1}^{lookback} r_{i,t-k}
       where we skip the most recent month to avoid short-term reversal

    2. Rank stocks cross-sectionally by momentum

    3. Construct dollar-neutral portfolio:
       - Long top decile (highest momentum)
       - Short bottom decile (lowest momentum)

    Key enhancements:
    - Volatility scaling: weight inversely by realized volatility
    - Winsorization: trim extreme returns to reduce noise
    - Sector neutralization: neutralize sector exposures
    """

    def __init__(self, config: Optional[MomentumConfig] = None):
        """
        Initialize the cross-sectional momentum signal generator.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or MomentumConfig()

        # State tracking
        self.price_history: Dict[str, pd.Series] = {}
        self.sector_map: Dict[str, str] = {}
        self.last_signals: Dict[str, float] = {}
        self.last_weights: Optional[PortfolioWeights] = None

        # Trade tracking
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.total_pnl: float = 0.0

    def update_prices(
        self,
        prices: Union[pd.DataFrame, Dict[str, pd.Series]],
        sectors: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Update price history for all stocks.

        Args:
            prices: DataFrame with tickers as columns and dates as index,
                   or dict mapping ticker to price series
            sectors: Optional dict mapping ticker to sector/industry
        """
        if isinstance(prices, pd.DataFrame):
            for ticker in prices.columns:
                self.price_history[ticker] = prices[ticker].dropna()
        else:
            for ticker, series in prices.items():
                self.price_history[ticker] = series.dropna()

        if sectors:
            self.sector_map.update(sectors)

    def calculate_returns(
        self,
        prices: pd.Series,
        log_returns: bool = True
    ) -> pd.Series:
        """
        Calculate returns from price series.

        Args:
            prices: Price series
            log_returns: If True, use log returns; else simple returns

        Returns:
            Returns series
        """
        if log_returns:
            return np.log(prices / prices.shift(1))
        else:
            return prices.pct_change()

    def winsorize_returns(
        self,
        returns: pd.Series,
        lower_pct: float = 0.01,
        upper_pct: float = 0.99
    ) -> pd.Series:
        """
        Winsorize extreme returns to reduce noise.

        Args:
            returns: Returns series
            lower_pct: Lower percentile for winsorization
            upper_pct: Upper percentile for winsorization

        Returns:
            Winsorized returns
        """
        lower_bound = returns.quantile(lower_pct)
        upper_bound = returns.quantile(upper_pct)
        return returns.clip(lower=lower_bound, upper=upper_bound)

    def calculate_momentum_signal(
        self,
        ticker: str,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> Optional[float]:
        """
        Calculate momentum signal for a single stock.

        The formula: mom_{i,t} = sum_{k=skip+1}^{lookback} r_{i,t-k}

        Args:
            ticker: Stock ticker
            as_of_date: Calculate momentum as of this date. If None, use latest.

        Returns:
            Momentum signal value, or None if insufficient data
        """
        if ticker not in self.price_history:
            return None

        prices = self.price_history[ticker]

        if as_of_date is not None:
            prices = prices[prices.index <= as_of_date]

        # Need enough history
        min_required = self.config.lookback_start + 5
        if len(prices) < min_required:
            return None

        # Calculate log returns
        returns = self.calculate_returns(prices, log_returns=True)

        # Winsorize if enabled
        if self.config.winsorize:
            returns = self.winsorize_returns(
                returns,
                lower_pct=self.config.winsorize_pct,
                upper_pct=1 - self.config.winsorize_pct
            )

        # Extract the momentum window: from t-lookback_start to t-lookback_end
        # Skip the most recent lookback_end days (typically 21 days / 1 month)
        lookback_returns = returns.iloc[-(self.config.lookback_start):]
        if len(lookback_returns) >= self.config.lookback_start:
            # Remove the most recent skip period
            momentum_returns = lookback_returns.iloc[:-self.config.lookback_end]
        else:
            # Not enough data
            return None

        # Calculate momentum based on type
        if self.config.momentum_type == MomentumType.CUMULATIVE:
            # Sum of log returns (equivalent to log of cumulative return)
            momentum = momentum_returns.sum()

        elif self.config.momentum_type == MomentumType.TOTAL_RETURN:
            # Cumulative return as percentage
            momentum = np.exp(momentum_returns.sum()) - 1

        elif self.config.momentum_type == MomentumType.RISK_ADJUSTED:
            # Sharpe-like: mean return / std of returns
            if momentum_returns.std() > 0:
                momentum = momentum_returns.mean() / momentum_returns.std()
            else:
                momentum = 0.0

        else:
            momentum = momentum_returns.sum()

        return momentum

    def calculate_volatility(
        self,
        ticker: str,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> Optional[float]:
        """
        Calculate realized volatility for a stock.

        Args:
            ticker: Stock ticker
            as_of_date: Calculate as of this date

        Returns:
            Annualized volatility, or None if insufficient data
        """
        if ticker not in self.price_history:
            return None

        prices = self.price_history[ticker]

        if as_of_date is not None:
            prices = prices[prices.index <= as_of_date]

        if len(prices) < self.config.vol_lookback + 1:
            return None

        returns = self.calculate_returns(prices, log_returns=True)
        recent_returns = returns.iloc[-self.config.vol_lookback:]

        volatility = recent_returns.std() * np.sqrt(self.config.annualization_factor)
        return volatility

    def generate_signals(
        self,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, float]:
        """
        Generate momentum signals for all stocks.

        Args:
            as_of_date: Calculate signals as of this date

        Returns:
            Dictionary mapping ticker to momentum signal
        """
        signals = {}
        volatilities = {}

        for ticker in self.price_history.keys():
            mom = self.calculate_momentum_signal(ticker, as_of_date)
            if mom is not None:
                signals[ticker] = mom

                if self.config.volatility_scale:
                    vol = self.calculate_volatility(ticker, as_of_date)
                    if vol is not None and vol > 0:
                        volatilities[ticker] = vol

        # Volatility scaling: divide by volatility for risk-adjusted signal
        if self.config.volatility_scale and volatilities:
            for ticker in list(signals.keys()):
                if ticker in volatilities:
                    signals[ticker] = signals[ticker] / volatilities[ticker]
                else:
                    # Remove stocks without volatility data
                    del signals[ticker]

        # Sector neutralization: demean within each sector
        if self.config.sector_neutralize and self.sector_map:
            signals = self._sector_neutralize_signals(signals)

        self.last_signals = signals
        return signals

    def _sector_neutralize_signals(
        self,
        signals: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Neutralize signals within each sector by demeaning.

        Args:
            signals: Raw momentum signals

        Returns:
            Sector-neutralized signals
        """
        # Group by sector
        sector_signals: Dict[str, Dict[str, float]] = {}

        for ticker, signal in signals.items():
            sector = self.sector_map.get(ticker, 'Unknown')
            if sector not in sector_signals:
                sector_signals[sector] = {}
            sector_signals[sector][ticker] = signal

        # Demean within each sector
        neutralized = {}
        for sector, sec_signals in sector_signals.items():
            if len(sec_signals) > 1:
                mean_signal = np.mean(list(sec_signals.values()))
                for ticker, signal in sec_signals.items():
                    neutralized[ticker] = signal - mean_signal
            else:
                # Single stock in sector, can't neutralize
                for ticker, signal in sec_signals.items():
                    neutralized[ticker] = signal

        return neutralized

    def construct_portfolio(
        self,
        signals: Optional[Dict[str, float]] = None,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> PortfolioWeights:
        """
        Construct dollar-neutral portfolio from signals.

        Portfolio construction:
        1. Rank stocks by momentum signal
        2. Long top percentile (highest momentum)
        3. Short bottom percentile (lowest momentum)
        4. Equal-weight within each leg
        5. Scale to be dollar-neutral

        Args:
            signals: Pre-computed signals. If None, generates new signals.
            as_of_date: Date for signal calculation

        Returns:
            PortfolioWeights object with long/short allocations
        """
        if signals is None:
            signals = self.generate_signals(as_of_date)

        if len(signals) == 0:
            return PortfolioWeights()

        # Convert to DataFrame for ranking
        signal_df = pd.DataFrame.from_dict(signals, orient='index', columns=['momentum'])
        signal_df = signal_df.sort_values('momentum', ascending=False)

        n_stocks = len(signal_df)
        n_long = max(
            self.config.min_stocks_per_side,
            int(n_stocks * self.config.long_percentile)
        )
        n_short = max(
            self.config.min_stocks_per_side,
            int(n_stocks * self.config.short_percentile)
        )

        # Handle case where we don't have enough stocks
        n_long = min(n_long, n_stocks // 2)
        n_short = min(n_short, n_stocks // 2)

        if n_long == 0 or n_short == 0:
            return PortfolioWeights()

        # Select long and short baskets
        long_tickers = signal_df.head(n_long).index.tolist()
        short_tickers = signal_df.tail(n_short).index.tolist()

        # Equal weight within each leg, scaled by max position constraint
        long_weight = min(1.0 / n_long, self.config.max_position_weight)
        short_weight = min(1.0 / n_short, self.config.max_position_weight)

        # Normalize to sum to 1 on each side
        long_total = long_weight * n_long
        short_total = short_weight * n_short

        long_weights = {t: long_weight / long_total for t in long_tickers}
        short_weights = {t: -short_weight / short_total for t in short_tickers}

        # Create result
        portfolio = PortfolioWeights(
            long_weights=long_weights,
            short_weights=short_weights,
            net_exposure=sum(long_weights.values()) + sum(short_weights.values()),
            gross_exposure=sum(long_weights.values()) - sum(short_weights.values()),
            long_count=n_long,
            short_count=n_short
        )

        self.last_weights = portfolio
        return portfolio

    def get_target_positions(
        self,
        portfolio: PortfolioWeights,
        capital: float
    ) -> Dict[str, float]:
        """
        Convert portfolio weights to dollar positions.

        Args:
            portfolio: Portfolio weights
            capital: Total capital to allocate

        Returns:
            Dictionary mapping ticker to dollar position (positive=long, negative=short)
        """
        positions = {}

        for ticker, weight in portfolio.long_weights.items():
            positions[ticker] = weight * capital

        for ticker, weight in portfolio.short_weights.items():
            positions[ticker] = weight * capital  # weight is already negative

        return positions

    def calculate_turnover(
        self,
        new_weights: PortfolioWeights,
        old_weights: Optional[PortfolioWeights] = None
    ) -> float:
        """
        Calculate portfolio turnover.

        Args:
            new_weights: New portfolio weights
            old_weights: Old portfolio weights. If None, use last_weights.

        Returns:
            Turnover as sum of absolute weight changes
        """
        if old_weights is None:
            old_weights = self.last_weights

        if old_weights is None:
            # First portfolio, full turnover
            return 2.0  # 100% long + 100% short

        # Combine all weights
        old_all = {**old_weights.long_weights, **old_weights.short_weights}
        new_all = {**new_weights.long_weights, **new_weights.short_weights}

        all_tickers = set(old_all.keys()) | set(new_all.keys())

        turnover = 0.0
        for ticker in all_tickers:
            old_w = old_all.get(ticker, 0.0)
            new_w = new_all.get(ticker, 0.0)
            turnover += abs(new_w - old_w)

        return turnover

    def get_signal_statistics(
        self,
        signals: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate statistics about the current signals.

        Args:
            signals: Signal values. If None, use last_signals.

        Returns:
            Dictionary with signal statistics
        """
        if signals is None:
            signals = self.last_signals

        if not signals:
            return {}

        signal_values = list(signals.values())

        return {
            'mean': np.mean(signal_values),
            'std': np.std(signal_values),
            'min': np.min(signal_values),
            'max': np.max(signal_values),
            'median': np.median(signal_values),
            'skew': pd.Series(signal_values).skew(),
            'kurtosis': pd.Series(signal_values).kurtosis(),
            'n_stocks': len(signal_values),
            'pct_positive': sum(1 for s in signal_values if s > 0) / len(signal_values)
        }

    def backtest_signal(
        self,
        prices: pd.DataFrame,
        rebalance_frequency: int = 21,
        transaction_cost: float = 0.001,
        sectors: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Backtest the momentum strategy.

        Args:
            prices: DataFrame with tickers as columns, dates as index
            rebalance_frequency: Days between rebalances
            transaction_cost: Transaction cost as fraction of trade value
            sectors: Optional sector mapping for neutralization

        Returns:
            DataFrame with backtest results
        """
        # Initialize
        self.update_prices(prices, sectors)

        dates = prices.index
        min_start = self.config.lookback_start + 10

        results = []
        current_weights: Optional[PortfolioWeights] = None
        cumulative_return = 1.0

        for i, date in enumerate(dates):
            if i < min_start:
                continue

            # Get current prices
            current_prices = prices.loc[date]

            # Rebalance check
            days_since_start = i - min_start
            should_rebalance = (days_since_start % rebalance_frequency == 0)

            if should_rebalance or current_weights is None:
                # Generate new signals and portfolio
                signals = self.generate_signals(as_of_date=date)
                new_weights = self.construct_portfolio(signals)

                # Calculate turnover
                turnover = self.calculate_turnover(new_weights, current_weights)

                current_weights = new_weights
            else:
                turnover = 0.0

            # Calculate daily returns
            if i > min_start and current_weights is not None:
                # Get previous prices
                prev_prices = prices.iloc[i-1]

                # Calculate portfolio return
                daily_returns = (current_prices - prev_prices) / prev_prices

                portfolio_return = 0.0
                for ticker, weight in current_weights.long_weights.items():
                    if ticker in daily_returns.index and not np.isnan(daily_returns[ticker]):
                        portfolio_return += weight * daily_returns[ticker]

                for ticker, weight in current_weights.short_weights.items():
                    if ticker in daily_returns.index and not np.isnan(daily_returns[ticker]):
                        portfolio_return += weight * daily_returns[ticker]

                # Subtract transaction costs on rebalance days
                if should_rebalance:
                    portfolio_return -= turnover * transaction_cost

                cumulative_return *= (1 + portfolio_return)

                results.append({
                    'date': date,
                    'portfolio_return': portfolio_return,
                    'cumulative_return': cumulative_return,
                    'n_long': current_weights.long_count,
                    'n_short': current_weights.short_count,
                    'turnover': turnover if should_rebalance else 0.0,
                    'gross_exposure': current_weights.gross_exposure
                })

        return pd.DataFrame(results).set_index('date')

    def get_performance_metrics(
        self,
        backtest_results: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results.

        Args:
            backtest_results: DataFrame from backtest_signal()

        Returns:
            Dictionary with performance metrics
        """
        returns = backtest_results['portfolio_return']

        # Annualized metrics
        ann_return = returns.mean() * self.config.annualization_factor
        ann_vol = returns.std() * np.sqrt(self.config.annualization_factor)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        # Drawdown analysis
        cumulative = backtest_results['cumulative_return']
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Win rate
        win_rate = (returns > 0).mean()

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(self.config.annualization_factor)
        sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

        # Turnover
        avg_turnover = backtest_results['turnover'].mean() * (self.config.annualization_factor / 21)

        return {
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_annual_turnover': avg_turnover,
            'total_return': cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0.0,
            'n_days': len(returns)
        }

    def get_current_exposures(
        self,
        portfolio: Optional[PortfolioWeights] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get current sector/industry exposures.

        Args:
            portfolio: Portfolio weights. If None, use last_weights.

        Returns:
            Dictionary with sector exposures
        """
        if portfolio is None:
            portfolio = self.last_weights

        if portfolio is None or not self.sector_map:
            return {}

        sector_exposure = {}

        for ticker, weight in portfolio.long_weights.items():
            sector = self.sector_map.get(ticker, 'Unknown')
            if sector not in sector_exposure:
                sector_exposure[sector] = {'long': 0.0, 'short': 0.0, 'net': 0.0}
            sector_exposure[sector]['long'] += weight
            sector_exposure[sector]['net'] += weight

        for ticker, weight in portfolio.short_weights.items():
            sector = self.sector_map.get(ticker, 'Unknown')
            if sector not in sector_exposure:
                sector_exposure[sector] = {'long': 0.0, 'short': 0.0, 'net': 0.0}
            sector_exposure[sector]['short'] += abs(weight)
            sector_exposure[sector]['net'] += weight  # weight is negative

        return sector_exposure

    def get_signal_decay_analysis(
        self,
        prices: pd.DataFrame,
        horizons: List[int] = [1, 5, 10, 21, 63],
        sectors: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Analyze signal decay / predictive power at different horizons.

        Args:
            prices: Price DataFrame
            horizons: Forward return horizons to analyze (in days)
            sectors: Optional sector mapping

        Returns:
            DataFrame with IC (information coefficient) at each horizon
        """
        self.update_prices(prices, sectors)

        dates = prices.index
        min_start = self.config.lookback_start + max(horizons) + 10

        ic_results = {h: [] for h in horizons}

        for i, date in enumerate(dates):
            if i < min_start:
                continue
            if i + max(horizons) >= len(dates):
                break

            # Generate signals
            signals = self.generate_signals(as_of_date=date)

            if len(signals) < 10:
                continue

            # Calculate forward returns at each horizon
            for horizon in horizons:
                future_date = dates[i + horizon]
                current_prices = prices.loc[date]
                future_prices = prices.loc[future_date]

                forward_returns = {}
                for ticker in signals.keys():
                    if ticker in current_prices.index and ticker in future_prices.index:
                        curr_p = current_prices[ticker]
                        fut_p = future_prices[ticker]
                        if not np.isnan(curr_p) and not np.isnan(fut_p) and curr_p > 0:
                            forward_returns[ticker] = (fut_p - curr_p) / curr_p

                # Calculate rank IC
                if len(forward_returns) >= 5:
                    common_tickers = [t for t in forward_returns.keys() if t in signals]
                    if len(common_tickers) >= 5:
                        signal_vals = [signals[t] for t in common_tickers]
                        return_vals = [forward_returns[t] for t in common_tickers]

                        # Spearman rank correlation
                        from scipy import stats
                        ic, _ = stats.spearmanr(signal_vals, return_vals)
                        ic_results[horizon].append({
                            'date': date,
                            'ic': ic
                        })

        # Aggregate results
        summary = []
        for horizon in horizons:
            if ic_results[horizon]:
                ics = [r['ic'] for r in ic_results[horizon]]
                summary.append({
                    'horizon': horizon,
                    'mean_ic': np.mean(ics),
                    'ic_std': np.std(ics),
                    'ic_ir': np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0,
                    'pct_positive': sum(1 for ic in ics if ic > 0) / len(ics),
                    'n_observations': len(ics)
                })

        return pd.DataFrame(summary)

    def reset(self) -> None:
        """Reset all state."""
        self.price_history = {}
        self.sector_map = {}
        self.last_signals = {}
        self.last_weights = None
        self.positions = {}
        self.trades = []
        self.total_pnl = 0.0
