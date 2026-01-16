"""
Volatility mean reversion strategy.

This strategy exploits the tendency of volatility to revert to its historical mean.
When implied volatility is significantly above the mean, sell options (short vega).
When implied volatility is significantly below the mean, buy options (long vega).
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ..utils.volatility import historical_volatility
from ..utils.black_scholes import black_scholes_price, calculate_greeks


class VolatilityMeanReversionStrategy:
    """
    Volatility mean reversion strategy.

    The strategy:
    1. Calculates mean and standard deviation of historical implied volatility
    2. Generates signals when current IV deviates significantly from mean
    3. Buy options when IV is low (below mean - threshold)
    4. Sell options when IV is high (above mean + threshold)
    5. Exit when volatility reverts to mean
    """

    def __init__(
        self,
        lookback_period: int = 30,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.5,
        transaction_cost: float = 0.001
    ):
        """
        Initialize the volatility mean reversion strategy.

        Args:
            lookback_period: Number of days to calculate vol statistics
            entry_threshold: Number of std devs from mean to enter
            exit_threshold: Number of std devs from mean to exit
            transaction_cost: Transaction cost as fraction of trade value
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost
        self.positions = []
        self.trades = []
        self.vol_history = []

    def update_volatility_statistics(
        self,
        implied_vol: float
    ) -> Dict[str, float]:
        """
        Update volatility history and calculate statistics.

        Args:
            implied_vol: Current implied volatility

        Returns:
            Dictionary with volatility statistics
        """
        self.vol_history.append(implied_vol)

        if len(self.vol_history) > self.lookback_period:
            self.vol_history.pop(0)

        if len(self.vol_history) < 10:  # Need minimum history
            return {
                'mean': np.nan,
                'std': np.nan,
                'z_score': np.nan,
                'upper_band': np.nan,
                'lower_band': np.nan
            }

        vol_array = np.array(self.vol_history)
        mean = np.mean(vol_array)
        std = np.std(vol_array)

        z_score = (implied_vol - mean) / std if std > 0 else 0

        return {
            'mean': mean,
            'std': std,
            'z_score': z_score,
            'upper_band': mean + self.entry_threshold * std,
            'lower_band': mean - self.entry_threshold * std,
            'exit_upper': mean + self.exit_threshold * std,
            'exit_lower': mean - self.exit_threshold * std
        }

    def generate_signal(
        self,
        implied_vol: float,
        vol_stats: Dict[str, float]
    ) -> str:
        """
        Generate trading signal based on volatility statistics.

        Args:
            implied_vol: Current implied volatility
            vol_stats: Volatility statistics from update_volatility_statistics

        Returns:
            Signal: 'BUY', 'SELL', 'CLOSE_LONG', 'CLOSE_SHORT', or 'HOLD'
        """
        if np.isnan(vol_stats['z_score']):
            return 'HOLD'

        z_score = vol_stats['z_score']

        # Entry signals
        if z_score < -self.entry_threshold:
            return 'BUY'  # IV is low, buy options
        elif z_score > self.entry_threshold:
            return 'SELL'  # IV is high, sell options

        # Exit signals for existing positions
        if len(self.positions) > 0:
            for pos in self.positions:
                if pos['direction'] == 'LONG' and z_score > -self.exit_threshold:
                    return 'CLOSE_LONG'
                elif pos['direction'] == 'SHORT' and z_score < self.exit_threshold:
                    return 'CLOSE_SHORT'

        return 'HOLD'

    def enter_long_position(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        quantity: int,
        option_price: float,
        vol_stats: Dict[str, float]
    ) -> Dict:
        """
        Enter a long volatility position (buy options).

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility
            option_type: 'call' or 'put'
            quantity: Number of contracts
            option_price: Price per option
            vol_stats: Current volatility statistics

        Returns:
            Trade details
        """
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)

        cost = option_price * quantity * 100
        transaction_cost = cost * self.transaction_cost
        total_cost = cost + transaction_cost

        position = {
            'direction': 'LONG',
            'S': S,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'option_type': option_type,
            'quantity': quantity,
            'entry_price': option_price,
            'entry_vol': sigma,
            'entry_z_score': vol_stats['z_score'],
            'cost': total_cost,
            'greeks': greeks
        }

        self.positions.append(position)

        trade = {
            'action': 'BUY',
            'stock_price': S,
            'strike': K,
            'option_type': option_type,
            'option_price': option_price,
            'quantity': quantity,
            'implied_vol': sigma,
            'z_score': vol_stats['z_score'],
            'cost': total_cost,
            'vega': greeks['vega'] * quantity * 100
        }

        self.trades.append(trade)
        return trade

    def enter_short_position(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        quantity: int,
        option_price: float,
        vol_stats: Dict[str, float]
    ) -> Dict:
        """
        Enter a short volatility position (sell options).

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility
            option_type: 'call' or 'put'
            quantity: Number of contracts
            option_price: Price per option
            vol_stats: Current volatility statistics

        Returns:
            Trade details
        """
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)

        proceeds = option_price * quantity * 100
        transaction_cost = proceeds * self.transaction_cost
        net_proceeds = proceeds - transaction_cost

        position = {
            'direction': 'SHORT',
            'S': S,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'option_type': option_type,
            'quantity': quantity,
            'entry_price': option_price,
            'entry_vol': sigma,
            'entry_z_score': vol_stats['z_score'],
            'proceeds': net_proceeds,
            'greeks': greeks
        }

        self.positions.append(position)

        trade = {
            'action': 'SELL',
            'stock_price': S,
            'strike': K,
            'option_type': option_type,
            'option_price': option_price,
            'quantity': quantity,
            'implied_vol': sigma,
            'z_score': vol_stats['z_score'],
            'proceeds': net_proceeds,
            'vega': -greeks['vega'] * quantity * 100  # Negative for short
        }

        self.trades.append(trade)
        return trade

    def close_position(
        self,
        position_index: int,
        S: float,
        option_price: float,
        current_vol: float,
        vol_stats: Dict[str, float]
    ) -> Dict:
        """
        Close an existing position.

        Args:
            position_index: Index of position to close
            S: Current stock price
            option_price: Current option price
            current_vol: Current implied volatility
            vol_stats: Current volatility statistics

        Returns:
            Trade details
        """
        if position_index >= len(self.positions):
            raise ValueError(f"Invalid position index: {position_index}")

        pos = self.positions[position_index]
        direction = pos['direction']

        if direction == 'LONG':
            # Closing long position (selling)
            proceeds = option_price * pos['quantity'] * 100
            transaction_cost = proceeds * self.transaction_cost
            net_proceeds = proceeds - transaction_cost
            pnl = net_proceeds - pos['cost']

        else:  # SHORT
            # Closing short position (buying back)
            cost = option_price * pos['quantity'] * 100
            transaction_cost = cost * self.transaction_cost
            total_cost = cost + transaction_cost
            pnl = pos['proceeds'] - total_cost

        vol_change = current_vol - pos['entry_vol']

        trade = {
            'action': f'CLOSE_{direction}',
            'stock_price': S,
            'strike': pos['K'],
            'option_type': pos['option_type'],
            'entry_price': pos['entry_price'],
            'exit_price': option_price,
            'quantity': pos['quantity'],
            'entry_vol': pos['entry_vol'],
            'exit_vol': current_vol,
            'vol_change': vol_change,
            'entry_z_score': pos['entry_z_score'],
            'exit_z_score': vol_stats['z_score'],
            'pnl': pnl
        }

        self.trades.append(trade)
        self.positions.pop(position_index)

        return trade

    def calculate_portfolio_vega(self, S: float) -> float:
        """
        Calculate total portfolio vega exposure.

        Args:
            S: Current stock price

        Returns:
            Total vega exposure
        """
        total_vega = 0.0

        for pos in self.positions:
            greeks = calculate_greeks(
                S, pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type']
            )
            position_vega = greeks['vega'] * pos['quantity'] * 100

            if pos['direction'] == 'SHORT':
                position_vega = -position_vega

            total_vega += position_vega

        return total_vega

    def get_performance_summary(self) -> Dict:
        """
        Get performance summary of the strategy.

        Returns:
            Dictionary with performance metrics
        """
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_pnl_per_trade': 0.0
            }

        closed_trades = [t for t in self.trades if 'pnl' in t]

        if len(closed_trades) == 0:
            return {
                'total_trades': len(self.trades),
                'closed_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_pnl_per_trade': 0.0
            }

        pnls = [t['pnl'] for t in closed_trades]
        total_pnl = sum(pnls)
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        win_rate = winning_trades / len(closed_trades) if closed_trades else 0

        return {
            'total_trades': len(self.trades),
            'closed_trades': len(closed_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_pnl_per_trade': total_pnl / len(closed_trades) if closed_trades else 0,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0
        }

    def get_trade_history(self) -> pd.DataFrame:
        """
        Get complete trade history as DataFrame.

        Returns:
            DataFrame with all trades
        """
        return pd.DataFrame(self.trades)
