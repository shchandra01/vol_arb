"""
Volatility Arbitrage Trading Framework.

This package provides tools for volatility-based trading strategies including:
- Options pricing and Greeks calculation
- Volatility estimation methods
- Trading strategies (delta-neutral, mean reversion, calendar spreads, etc.)
- Alpha signals (cross-sectional momentum, etc.)
"""

from .signals import CrossSectionalMomentum

__all__ = ['CrossSectionalMomentum']

__version__ = '0.1.0'
