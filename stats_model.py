import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from typing import Generator, Tuple, List
import statsmodels.api as sm

def calculate_spread(btc_price: np.ndarray, alt_price: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Perform OLS regression: ln(P_btc) = alpha + beta * ln(P_alt) + epsilon.
    :return: (spread_series, beta, alpha)
    """
    # 1. Prepare the independent variable (X)
    # Add a constant term to the alt_price to estimate the intercept 'alpha'
    X = sm.add_constant(alt_price)
    
    # 2. Define the dependent variable (y)
    y = btc_price
    
    # 3. Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(y, X).fit()
    
    # 4. Extract parameters
    # model.params contains [alpha, beta]
    alpha = model.params[0]
    beta = model.params[1]
    
    # 5. Calculate the residuals (spread)
    # epsilon = y - (alpha + beta * X)
    spread_series = model.resid
    
    return spread_series, beta, alpha

def kss_test(spread: np.ndarray, max_lag: int, critical_value: float = -2.66) -> Tuple[float, int, bool]:
    """
    Perform KSS nonlinear unit root test with automatic lag selection via AIC.
    The auxiliary regression: Δy_t = γ * y_{t-1}^3 + Σ ρ_j * Δy_{t-j} + error
    
    :param spread: The residual series (epsilon) from calculate_spread.
    :param max_lag: Maximum lags of Δy_t to include in the regression.
    :param critical_value: The CV for 10% significance (default -1.92 for Case 1, T=500).
    :return: A tuple of (t_stat, best_lag, is_stationary).
    """
    
    # 1. Prepare the basic variables
    # Δy_t (First difference)
    diff_y = np.diff(spread)
    # y_{t-1}^3 (Cubic term of the lagged level)
    y_lag_cubic = np.power(spread[:-1], 3)
    
    best_aic = np.inf
    best_t_stat = 0.0
    best_lag = 0

    # 2. Iterate through lags from 0 to max_lag to find the best AIC
    for p in range(max_lag + 1):
        # Prepare the regressors (Independent variables)
        # First regressor is always y_{t-1}^3
        X = y_lag_cubic.copy()
        
        # Target variable Δy_t starts from p+1 to align with lags
        y_target = diff_y[p:]
        X_current = X[p:]
        
        # If p > 0, add lagged difference terms: Δy_{t-1}, ..., Δy_{t-p}
        if p > 0:
            lagged_diffs = []
            for j in range(1, p + 1):
                # Slice diff_y to align with y_target
                lagged_diffs.append(diff_y[p-j:-j])
            
            # Combine y_{t-1}^3 and lagged differences
            X_combined = np.column_stack([X_current] + lagged_diffs)
        else:
            X_combined = X_current

        # 3. Fit OLS regression (No constant because Case 2 means data is de-meaned)
        # Note: In Case 2 of KSS, the auxiliary regression itself doesn't need 
        # an intercept if the original series was already de-meaned.
        model = sm.OLS(y_target, X_combined).fit()
        
        # 4. Update the best model based on AIC
        if model.aic < best_aic:
            best_aic = model.aic
            # The t-statistic of the first coefficient (gamma)
            best_t_stat = model.tvalues[0]
            best_lag = p

    # 5. Determine stationarity
    # If t_stat < critical_value, reject null hypothesis (unit root exists)
    is_stationary = best_t_stat < critical_value
    
    return best_t_stat, best_lag, is_stationary

def adf_test(spread: np.ndarray, max_lag: int, p_threshold: float = 0.1) -> Tuple[float, float, int, bool]:
    """
    Perform Augmented Dickey-Fuller (ADF) test with automatic lag selection using AIC.
    
    :param spread: The residual series (epsilon) from calculate_spread.
    :param max_lag: The upper limit for the number of lags to be checked.
    :param p_threshold: Significance level threshold (default 0.1).
    :return: A tuple of (adf_stat, p_value, used_lag, is_stationary).
    """
    
    # 1. Execute the ADF test with automatic lag selection
    # regression='c': Includes a constant in the test regression.
    # autolag='AIC': Automatically determines the lag length that minimizes the AIC.
    # maxlag: The maximum lag to consider.
    result = adfuller(spread, maxlag=max_lag, autolag='AIC', regression='c')
    
    # 2. Unpack the results
    # adfuller output format: (adf_stat, pvalue, usedlag, nobs, critical_values, icbest)
    adf_stat = result[0]
    p_value = result[1]
    used_lag = result[2]
    
    # 3. Determine stationarity
    # Null Hypothesis (H0): The series has a unit root (non-stationary).
    # If p-value < threshold, we reject H0.
    is_stationary = p_value < p_threshold
    
    return adf_stat, p_value, used_lag, is_stationary

def get_kendall_tau(price_a: np.ndarray, price_btc: np.ndarray) -> float:
    """
    Calculate Kendall's tau between asset price and BTC price.
    :return: Correlation coefficient.
    """
    # 1. Calculate Kendall's tau
    # kendalltau returns a tuple: (correlation, p_value)
    # We only need the correlation coefficient
    tau, _ = kendalltau(price_a, price_btc)
    
    # 2. Handle cases where tau might be NaN (e.g., zero variance in input)
    if np.isnan(tau):
        return 0.0
        
    return float(tau)

def screen_assets_with_spreads(
    train_data: pd.DataFrame, 
    btc_col: str = 'BTC', 
    max_lag: int = 6, 
    adf_threshold: float = 0.1, 
    kss_threshold: float = -1.92
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Screen assets by testing stationarity and correlation, and return 
    both the statistical summary and the combined spreads dataframe.
    
    :param train_data: DataFrame containing log-prices and [timestamp, datetime_utc].
    :param btc_col: The benchmark asset column name.
    :param max_lag: Max lag for AIC selection in tests.
    :param adf_threshold: P-value threshold for ADF.
    :param kss_threshold: Critical value for KSS.
    :return: A tuple containing:
             1. screening_df: Summary of stats for each asset.
             2. spreads_df: Time-series of spreads for all 7 assets.
    """
    
    # 1. Identify altcoin columns
    exclude_cols = [btc_col, 'timestamp', 'datetime_utc']
    alt_cols = [col for col in train_data.columns if col in ['BCH', 'LINK', 'ADA', 'XRP', 'ETH', 'ETC', 'LTC']]
    # If the specific 7 are not found, fallback to all other columns
    if not alt_cols:
        alt_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    # 2. Initialize the spreads DataFrame with time columns
    spreads_df = train_data[['timestamp', 'datetime_utc']].copy()
    
    analysis_results = []
    btc_prices = train_data[btc_col].values

    # 3. Iterate through each altcoin
    for alt_col in alt_cols:
        alt_prices = train_data[alt_col].values
        
        # --- Step A: Calculate Spread (OLS) ---
        spread, beta, alpha = calculate_spread(btc_prices, alt_prices)
        
        # Store the spread into the spreads_df
        spreads_df[alt_col] = spread
        
        # --- Step B: ADF Test ---
        adf_stat, adf_p, adf_lag, adf_pass = adf_test(
            spread, 
            max_lag=max_lag, 
            p_threshold=adf_threshold
        )
        
        # --- Step C: KSS Test ---
        kss_stat, kss_lag, kss_pass = kss_test(
            spread, 
            max_lag=max_lag, 
            critical_value=kss_threshold
        )
        
        # --- Step D: Kendall's Tau ---
        tau = get_kendall_tau(alt_prices, btc_prices)
        
        # --- Step E: Collect results ---
        analysis_results.append({
            'Asset': alt_col,
            'Beta': beta,
            'Alpha': alpha,
            'ADF_Stat': adf_stat,
            'ADF_PValue': adf_p,
            'ADF_Lag': adf_lag,
            'ADF_Pass': adf_pass,
            'KSS_Stat': kss_stat,
            'KSS_Lag': kss_lag,
            'KSS_Pass': kss_pass,
            'Kendall_Tau': tau,
            'Qualified': adf_pass or kss_pass
        })

    # 4. Create summary DataFrame and rank
    screening_df = pd.DataFrame(analysis_results)
    screening_df = screening_df.sort_values(by='Kendall_Tau', ascending=False).reset_index(drop=True)
    
    return screening_df, spreads_df