import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from itertools import combinations
from typing import List, Dict, Any
from stats_model import screen_assets_with_spreads
from copula_engine import get_pseudo_observations, fit_best_copula, calculate_h_index
from data_loader import get_rolling_windows

def select_pairs(train_data: pd.DataFrame, n_top: int = 2, max_lag: int = 6) -> List[Dict]:
    """
    Screen assets, rank them by Kendall tau, and form all unique combinations 
    of the top N qualified assets.
    """
    # 1. Get statistical summary and residuals against BTC to identify 'Qualified' assets
    summary_df, spread_df = screen_assets_with_spreads(train_data, max_lag=max_lag)
    
    # 2. Filter for qualified assets and take the top N based on Kendall Tau
    qualified_assets = summary_df[summary_df['Qualified'] == True].head(n_top)
    asset_names = qualified_assets['Asset'].tolist()
    
    # If fewer than 2 assets qualify, we cannot form a pair
    if len(asset_names) < 2:
        return []
    
    # 3. Form all unique pairs (Combinations)
    pair_combinations = list(combinations(asset_names, 2))
    
    pairs_info = []
    for asset_a, asset_b in pair_combinations:
        spread_a = spread_df[asset_a].values
        spread_b = spread_df[asset_b].values
        alpha_a = summary_df.loc[summary_df['Asset'] == asset_a, 'Alpha'].values[0]
        alpha_b = summary_df.loc[summary_df['Asset'] == asset_b, 'Alpha'].values[0]
        beta_a = summary_df.loc[summary_df['Asset'] == asset_a, 'Beta'].values[0]
        beta_b = summary_df.loc[summary_df['Asset'] == asset_b, 'Beta'].values[0]
        
        # Convert prices to pseudo-observations [0, 1] for Copula fitting
        u = get_pseudo_observations(spread_a)
        v = get_pseudo_observations(spread_b)
        
        # Fit the best Copula family for this specific pair
        best_copula = fit_best_copula(u, v)
        
        pairs_info.append({
            'asset_a': asset_a,
            'asset_b': asset_b,
            'alpha': [alpha_a, alpha_b],
            'beta': [beta_a, beta_b],
            'train_spread': [spread_a, spread_b],
            'copula_type': best_copula['type'],
            'theta': best_copula['theta'],
        })
        
    return pairs_info

def get_ecdf_transformer(train_data):
    sorted_data = np.sort(train_data)
    y = np.linspace(1/(len(train_data)+1), len(train_data)/(len(train_data)+1), len(train_data))
    # Return a function that maps new data to ranks
    return interp1d(sorted_data, y, kind='linear', bounds_error=False, fill_value=(0, 1))

def execute_trading_strategy(
    test_data: pd.DataFrame, 
    pairs_info: List[Dict], 
    alpha1: float =  0.1, 
    alpha2: float = 0.1
) -> pd.DataFrame:
    """
    Simulate trading for the test period. Signals are derived from the 
    conditional probabilities of the copula fitted to the pair (A, B).
    """
    test_results = test_data[['timestamp', 'datetime_utc']].copy()
    test_results['strategy_return'] = 0.0
    
    if not pairs_info:
        return test_results # Return zero returns if no pairs traded

    # Pre-calculate log returns for the test period for all involved assets
    df_log_returns = test_data.copy()
    target_cols = test_data.columns.difference(['timestamp', 'datetime_utc'])
    df_log_returns[target_cols] = test_data[target_cols].diff()
    
    pair_count = len(pairs_info)
    
    for pair in pairs_info:
        a = pair['asset_a']
        b = pair['asset_b']
        alpha = pair['alpha']
        beta = pair['beta']
        test_spread_a = (test_data['BTC'] - (beta[0] * test_data[a] + alpha[0])).values
        test_spread_b = (test_data['BTC'] - (beta[1] * test_data[b] + alpha[1])).values
        
        # Map test prices to (0, 1) using ECDF of the training set
        transformer_a = get_ecdf_transformer(pair['train_spread'][0])
        transformer_b = get_ecdf_transformer(pair['train_spread'][1])
        u_test = transformer_a(test_spread_a)
        v_test = transformer_b(test_spread_b)
        
        current_pos = 0 # 0: Flat, 1: Long A / Short B, -1: Short A / Long B
        pair_returns = np.zeros(len(test_data))
        
        for t in range(1, len(test_data)):
            # Calculate h-indices: h(u|v) and h(v|u)
            h_uv = calculate_h_index(u_test[t-1], v_test[t-1], pair['theta'], pair['copula_type'], '1|2')
            h_vu = calculate_h_index(u_test[t-1], v_test[t-1], pair['theta'], pair['copula_type'], '2|1')
            
            # --- SIGNAL LOGIC ---
            if h_uv < alpha1 and h_vu > (1 - alpha1):
                current_pos = 1
            elif h_uv > (1 - alpha1) and h_vu < alpha1:
                current_pos = -1
            elif (0.5 - alpha2) < h_uv < (0.5 + alpha2) and (0.5 - alpha2) < h_vu < (0.5 + alpha2):
                current_pos = 0
            
            # --- PNL CALCULATION ---
            if t > 0:
                ret_a = df_log_returns[a].iloc[t]
                ret_b = df_log_returns[b].iloc[t]
                pair_returns[t] = current_pos * (- beta[0] * ret_a + beta[1] * ret_b) / 2

        # Average returns across all active pairs
        test_results['strategy_return'] += (pair_returns / pair_count)
        
    return test_results

def run_full_backtest(raw_data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Rolling window controller. Iterates through the dataset week-by-week.
    """
    all_period_results = []
    
    # Assuming get_rolling_windows is available from your data_loader
    windows = get_rolling_windows(
        raw_data, 
        train_hours=config.get('train_hours', 504), 
        test_hours=config.get('test_hours', 168)
    )
    
    for train_df, test_df in windows:
        # 1. Pair selection and fitting for the current training window
        pairs = select_pairs(
            train_df, 
            n_top=config.get('n_top', 2), 
            max_lag=config.get('max_lag', 6)
        )
        
        # 2. Execute strategy on the following test week
        period_results = execute_trading_strategy(
            test_df, 
            pairs, 
            alpha1=config.get('alpha1', 0.1), 
            alpha2=config.get('alpha2', 0.1)
        )
        
        all_period_results.append(period_results)
        
    if not all_period_results:
        return pd.DataFrame()
        
    final_df = pd.concat(all_period_results).reset_index(drop=True)
    final_df['cumulative_return'] = final_df['strategy_return'].cumsum()
    
    return final_df