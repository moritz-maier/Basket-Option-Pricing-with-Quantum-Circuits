"""
Utility constants and small helpers for data generation.
"""

# Number of trading days per year (used for:
#   - annualizing volatility
#   - converting trading days to years in option pricing
#   - default maturity definitions
TRADING_DAYS = 252