"""
Black-Scholes option pricing model and Greeks calculations.
"""
import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> float:
    """
    Calculate Black-Scholes option price.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'

    Returns:
        Option price
    """
    if T <= 0:
        if option_type.lower() == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> Dict[str, float]:
    """
    Calculate all Greeks for an option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'

    Returns:
        Dictionary containing delta, gamma, theta, vega, and rho
    """
    if T <= 0:
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)

    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega (same for calls and puts, divided by 100 for 1% change)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Theta (per day, so divided by 365)
    if option_type.lower() == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Rho (divided by 100 for 1% change)
    if option_type.lower() == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = 'call',
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.

    Args:
        option_price: Market price of the option
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        option_type: 'call' or 'put'
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        Implied volatility
    """
    if T <= 0:
        return np.nan

    # Initial guess
    sigma = 0.3

    for i in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega = calculate_greeks(S, K, T, r, sigma, option_type)['vega']

        # Vega is per 1% change, so multiply by 100
        vega = vega * 100

        diff = option_price - price

        if abs(diff) < tolerance:
            return sigma

        if vega < 1e-10:  # Avoid division by zero
            break

        # Newton-Raphson update
        sigma = sigma + diff / vega

        # Keep sigma positive and reasonable
        sigma = max(0.001, min(sigma, 5.0))

    return np.nan


def delta_neutral_hedge_ratio(
    option_delta: float,
    option_quantity: int
) -> float:
    """
    Calculate the number of shares needed to delta hedge an option position.

    Args:
        option_delta: Delta of the option
        option_quantity: Number of option contracts (positive for long, negative for short)

    Returns:
        Number of shares to hold (negative means short)
    """
    # Each option contract is typically 100 shares
    shares_per_contract = 100
    return -option_delta * option_quantity * shares_per_contract


def gamma_scalping_pnl(
    initial_stock_price: float,
    final_stock_price: float,
    gamma: float,
    position_size: int
) -> float:
    """
    Estimate P&L from gamma scalping.

    Args:
        initial_stock_price: Stock price at start
        final_stock_price: Stock price at end
        gamma: Gamma of the option position
        position_size: Number of option contracts

    Returns:
        Estimated P&L from gamma scalping
    """
    shares_per_contract = 100
    price_change = final_stock_price - initial_stock_price

    # Gamma P&L = 0.5 * Gamma * (Price Change)^2
    gamma_pnl = 0.5 * gamma * (price_change ** 2) * position_size * shares_per_contract

    return gamma_pnl


def calculate_volatility_exposure(
    vega: float,
    position_size: int
) -> float:
    """
    Calculate total volatility exposure (vega) of a position.

    Args:
        vega: Vega per option (per 1% change in volatility)
        position_size: Number of option contracts

    Returns:
        Total vega exposure
    """
    shares_per_contract = 100
    return vega * position_size * shares_per_contract
