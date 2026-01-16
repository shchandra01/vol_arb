"""
Delta-neutral volatility trading strategy.

This strategy profits from volatility changes while maintaining a delta-neutral position.
The key is to buy options when implied volatility is low and hedge with the underlying.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ..utils.black_scholes import (
    black_scholes_price,
    calculate_greeks,
    implied_volatility,
    delta_neutral_hedge_ratio
)


class DeltaNeutralStrategy:
    """
    Delta-neutral volatility trading strategy.

    The strategy:
    1. Identifies options with low implied volatility relative to historical volatility
    2. Buys options (straddle or strangle) to get long vega
    3. Hedges delta to zero by taking opposite position in underlying
    4. Rebalances periodically to maintain delta neutrality
    5. Profits from increase in volatility and/or realized volatility through gamma scalping
    """

    def __init__(
        self,
        rebalance_threshold: float = 0.1,
        transaction_cost: float = 0.001
    ):
        """
        Initialize the delta-neutral strategy.

        Args:
            rebalance_threshold: Delta threshold to trigger rebalancing (e.g., 0.1 = 10 delta)
            transaction_cost: Transaction cost as fraction of trade value
        """
        self.rebalance_threshold = rebalance_threshold
        self.transaction_cost = transaction_cost
        self.positions = []
        self.hedge_shares = 0
        self.total_pnl = 0.0
        self.trades = []

    def analyze_opportunity(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        implied_vol: float,
        historical_vol: float,
        option_type: str = 'call'
    ) -> Dict:
        """
        Analyze whether there's a volatility arbitrage opportunity.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            implied_vol: Implied volatility from market
            historical_vol: Historical volatility
            option_type: 'call' or 'put'

        Returns:
            Dictionary with opportunity analysis
        """
        vol_spread = implied_vol - historical_vol
        vol_ratio = implied_vol / historical_vol if historical_vol > 0 else np.inf

        # Calculate option price at both volatilities
        price_implied = black_scholes_price(S, K, T, r, implied_vol, option_type)
        price_historical = black_scholes_price(S, K, T, r, historical_vol, option_type)

        # Calculate Greeks
        greeks = calculate_greeks(S, K, T, r, implied_vol, option_type)

        opportunity_score = -vol_spread  # Negative spread means IV < HV (good for buying)

        return {
            'opportunity_score': opportunity_score,
            'vol_spread': vol_spread,
            'vol_ratio': vol_ratio,
            'implied_vol': implied_vol,
            'historical_vol': historical_vol,
            'price_implied': price_implied,
            'price_historical': price_historical,
            'fair_value_edge': price_historical - price_implied,
            'greeks': greeks,
            'recommendation': 'BUY' if vol_spread < -0.05 else 'PASS'
        }

    def enter_position(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        quantity: int,
        option_price: float
    ) -> Dict:
        """
        Enter a new option position and establish delta hedge.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            quantity: Number of contracts (positive for long)
            option_price: Price paid per option

        Returns:
            Trade details
        """
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)

        position = {
            'S': S,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'option_type': option_type,
            'quantity': quantity,
            'entry_price': option_price,
            'entry_stock_price': S,
            'greeks': greeks
        }

        self.positions.append(position)

        # Calculate hedge ratio
        shares_to_hedge = delta_neutral_hedge_ratio(greeks['delta'], quantity)

        # Execute hedge
        self.hedge_shares += shares_to_hedge
        hedge_cost = abs(shares_to_hedge) * S * self.transaction_cost

        # Calculate total cost
        option_cost = option_price * quantity * 100  # 100 shares per contract
        total_cost = option_cost + hedge_cost

        trade = {
            'action': 'ENTER',
            'stock_price': S,
            'option_price': option_price,
            'quantity': quantity,
            'hedge_shares': shares_to_hedge,
            'cost': total_cost,
            'portfolio_delta': self.calculate_portfolio_delta(S)
        }

        self.trades.append(trade)
        self.total_pnl -= total_cost

        return trade

    def calculate_portfolio_delta(self, S: float) -> float:
        """
        Calculate total portfolio delta.

        Args:
            S: Current stock price

        Returns:
            Total portfolio delta
        """
        total_delta = 0.0

        for pos in self.positions:
            # Recalculate greeks at current price
            greeks = calculate_greeks(
                S, pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type']
            )
            option_delta = greeks['delta'] * pos['quantity'] * 100

            total_delta += option_delta

        # Add hedge delta (1 delta per share)
        total_delta += self.hedge_shares

        return total_delta

    def rebalance_hedge(
        self,
        S: float,
        current_time: float = 0.0
    ) -> Optional[Dict]:
        """
        Rebalance delta hedge if needed.

        Args:
            S: Current stock price
            current_time: Current time (for updating time to expiration)

        Returns:
            Rebalance trade details if rebalance occurred, None otherwise
        """
        # Update positions with new time to expiration
        for pos in self.positions:
            time_passed = current_time
            pos['T'] = max(0, pos['T'] - time_passed)

        portfolio_delta = self.calculate_portfolio_delta(S)

        if abs(portfolio_delta) > self.rebalance_threshold * 100:
            # Need to rebalance
            shares_to_trade = -portfolio_delta

            trade_cost = abs(shares_to_trade) * S * self.transaction_cost
            self.hedge_shares += shares_to_trade
            self.total_pnl -= trade_cost

            # Calculate P&L from hedge rebalance (gamma scalping)
            rebalance_pnl = 0.0
            for pos in self.positions:
                price_change = S - pos['entry_stock_price']
                greeks = calculate_greeks(
                    S, pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type']
                )
                gamma_pnl = 0.5 * greeks['gamma'] * (price_change ** 2) * pos['quantity'] * 100
                rebalance_pnl += gamma_pnl

            trade = {
                'action': 'REBALANCE',
                'stock_price': S,
                'portfolio_delta': portfolio_delta,
                'shares_traded': shares_to_trade,
                'new_hedge_shares': self.hedge_shares,
                'cost': trade_cost,
                'gamma_pnl': rebalance_pnl,
                'new_delta': self.calculate_portfolio_delta(S)
            }

            self.trades.append(trade)
            return trade

        return None

    def mark_to_market(
        self,
        S: float,
        current_vols: Dict[int, float]
    ) -> Dict:
        """
        Mark all positions to market.

        Args:
            S: Current stock price
            current_vols: Dictionary mapping position index to current implied volatility

        Returns:
            P&L summary
        """
        total_option_value = 0.0

        for i, pos in enumerate(self.positions):
            sigma = current_vols.get(i, pos['sigma'])
            current_price = black_scholes_price(
                S, pos['K'], pos['T'], pos['r'], sigma, pos['option_type']
            )
            position_value = current_price * pos['quantity'] * 100
            total_option_value += position_value

        # Hedge value
        hedge_value = self.hedge_shares * S

        # Total portfolio value
        total_value = total_option_value + hedge_value * (1 if self.hedge_shares < 0 else -1)

        return {
            'total_option_value': total_option_value,
            'hedge_value': abs(hedge_value),
            'net_value': total_value,
            'realized_pnl': self.total_pnl,
            'unrealized_pnl': total_value,
            'total_pnl': self.total_pnl + total_value
        }

    def close_position(
        self,
        S: float,
        position_index: int,
        option_price: float
    ) -> Dict:
        """
        Close a specific position.

        Args:
            S: Current stock price
            position_index: Index of position to close
            option_price: Current market price of option

        Returns:
            Trade details
        """
        if position_index >= len(self.positions):
            raise ValueError(f"Invalid position index: {position_index}")

        pos = self.positions[position_index]

        # Close option position
        proceeds = option_price * pos['quantity'] * 100
        transaction_cost = proceeds * self.transaction_cost

        # Unwind hedge
        greeks = calculate_greeks(S, pos['K'], pos['T'], pos['r'], pos['sigma'], pos['option_type'])
        shares_to_unwind = delta_neutral_hedge_ratio(greeks['delta'], pos['quantity'])
        self.hedge_shares -= shares_to_unwind
        hedge_cost = abs(shares_to_unwind) * S * self.transaction_cost

        # Calculate P&L
        option_pnl = (option_price - pos['entry_price']) * pos['quantity'] * 100
        total_pnl = proceeds - transaction_cost - hedge_cost

        self.total_pnl += total_pnl

        trade = {
            'action': 'CLOSE',
            'stock_price': S,
            'option_price': option_price,
            'entry_price': pos['entry_price'],
            'quantity': pos['quantity'],
            'option_pnl': option_pnl,
            'total_pnl': total_pnl,
            'remaining_positions': len(self.positions) - 1
        }

        self.trades.append(trade)
        self.positions.pop(position_index)

        return trade

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all trades.

        Returns:
            DataFrame with trade history
        """
        return pd.DataFrame(self.trades)
