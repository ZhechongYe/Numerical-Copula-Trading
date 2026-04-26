import pandas as pd
import glob
import os
import numpy as np
from typing import Generator, Tuple, List

def load_and_preprocess_data(folder_path: str = "data/1h/") -> pd.DataFrame:
    """
    Load multiple cryptocurrency CSV files, merge them by timestamp, 
    and convert closing prices to log-prices.
    
    :param folder_path: Path to the directory containing xxx_USDT_okx.csv files.
    :return: A merged DataFrame with columns [timestamp, datetime_utc, Asset1, Asset2, ...].
             Values for assets are in log scale (ln(P)).
    """
    # 1. Find all CSV files matching the pattern
    file_pattern = os.path.join(folder_path, "*_USDT_okx.csv")
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        raise FileNotFoundError(f"No CSV files found in {folder_path}. Please check the directory path.")

    combined_df = None

    for path in file_paths:
        # Extract asset symbol from filename (e.g., 'BTC_USDT_okx.csv' -> 'BTC')
        file_name = os.path.basename(path)
        asset_label = file_name.split('_')[0]
        
        # 2. Read CSV and keep only necessary columns
        # Expected columns in CSV: timestamp, datetime_utc, close, ...
        df = pd.read_csv(path)
        df = df[['timestamp', 'datetime_utc', 'close']].copy()
        
        # 3. Rename 'close' column to the specific asset symbol
        df = df.rename(columns={'close': asset_label})
        
        # 4. Merge dataframes on time columns
        if combined_df is None:
            combined_df = df
        else:
            # Use outer join to ensure all timestamps are captured and aligned
            combined_df = pd.merge(combined_df, df, on=['timestamp', 'datetime_utc'], how='outer')

    # 5. Sort by timestamp to ensure chronological order
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

    # 6. Handle potential missing values 
    # Forward fill is used to handle brief gaps in exchange data
    combined_df = combined_df.ffill().dropna()

    # 7. Apply Log Transformation to all asset columns
    # Exclude non-price columns like 'timestamp' and 'datetime_utc'
    numeric_cols = [col for col in combined_df.columns if col not in ['timestamp', 'datetime_utc']]
    for col in numeric_cols:
        combined_df[col] = np.log(combined_df[col])

    print(f"Data loading complete. Assets processed: {numeric_cols}")
    return combined_df

def get_rolling_windows(data: pd.DataFrame, train_hours: int = 504, test_hours: int = 168) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Generate rolling windows for training and testing.
    :param data: Preprocessed log price DataFrame.
    :param train_hours: Number of hours for formation period (e.g., 3 weeks = 504).
    :param test_hours: Number of hours for trading period (e.g., 1 week = 168).
    :return: Generator yielding (train_df, test_df).
    """
    # Calculate the total length required for one full cycle (3 weeks + 1 week)
    total_window_size = train_hours + test_hours
    total_len = len(data)
    
    # We slide the window with a stride of 'test_hours'.
    # This means after the first 'test_hours' are traded, the next window 
    # moves forward by exactly 1 week to start the next trade.
    for start_idx in range(0, total_len - total_window_size + 1, test_hours):
        
        # 1. Define indices for the training (formation) set
        train_start = start_idx
        train_end = start_idx + train_hours
        
        # 2. Define indices for the testing (trading) set
        # The test set starts immediately after the training set ends
        test_start = train_end
        test_end = train_end + test_hours
        
        # 3. Extract the dataframes
        # .iloc is used for integer-location based indexing (exclusive of the stop index)
        train_df = data.iloc[train_start:train_end].copy()
        test_df = data.iloc[test_start:test_end].copy()
        
        # 4. Optional: Safety check to ensure windows are valid
        if len(train_df) == train_hours and len(test_df) == test_hours:
            yield train_df, test_df
        else:
            # Stop if we don't have enough data to fill a complete test window
            break