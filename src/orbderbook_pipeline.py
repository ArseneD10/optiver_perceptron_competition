import polars as pl 
import numpy as np
from torch.utils.data import TensorDataset
import torch
from sklearn.model_selection import train_test_split

def run_lobster_pipeline(book: pl.DataFrame, 
                         trade: pl.DataFrame, 
                         window_size: int | str,
                         test_size: float,
                         valid_size: float,
                         to_tensor_dataset: bool = True):
    df = pl.concat([book, trade], how='horizontal_extend').sort("Time")
    df = df.filter((pl.col("Type") == 4) | (pl.col("Type") == 5)) # select only execution type
    df, ror_index = prepare_book(df)
    df = df.drop_nulls()
    print(df.null_count())

    if isinstance(window_size, int):
        X = bar_aggregation(df, window_size=window_size)
    
    train_tmp, X_test = train_test_split(X, test_size=test_size)
    X_train, X_valid = train_test_split(train_tmp, test_size=valid_size)

    if to_tensor_dataset:
        X_train = TensorDataset(torch.from_numpy(X_train).float())
        X_valid = TensorDataset(torch.from_numpy(X_valid).float())
        X_test = TensorDataset(torch.from_numpy(X_test).float())

    return X_train, X_valid, X_test, ror_index


def prepare_book(df: pl.DataFrame, cov_window: int=30):
    df = mid_price(df)
    df = mid_return(df)
    df = ob_imbalance(df)
    df = spread(df)
    df = trade_imbalance(df)
    df = get_lagged_cov(df, "ror_mid", "imbalance", cov_window)
    return df.select(
        ["ror_mid",
         "TradeImbalance",
         'imbalance',
         "cov_ror_mid_imbalance",
         "spread"
         ]
    ), 0

def prepare_book_for_phase_2(book: pl.DataFrame, 
                         trade: pl.DataFrame, 
                         cum_sum_window_size: int = 10,
                         trade_imbalance_window_size: int = 10):
    
    df = pl.concat([book, trade], how='horizontal_extend').sort("Time")
    df = df.filter((pl.col("Type") == 4) | (pl.col("Type") == 5)) # select only execution type
    df = mid_price(df)
    df = mid_return(df)
    df = cumulative_return(df, cum_sum_window_size)
    df = ob_imbalance(df)
    df = trade_imbalance(df, window_size=trade_imbalance_window_size)
    return df.select([
        "ror_mid",
        "cum_return",
        "TradeImbalance",
        "imbalance",
        "Time"
    ])

def mid_price(df: pl.DataFrame):
    return df.with_columns(
        ((pl.col("ask_price_1")+pl.col("bid_price_1")) / 2).alias("mid")
    )

def trade_imbalance(df: pl.DataFrame, window_size: int = 30):
    return df.with_columns(
        ((pl.col("Direction") + 1) / 2).alias("NormalizedDirection")
    ).with_columns(
        pl.col("NormalizedDirection").rolling_mean(window_size=window_size).alias("TradeImbalance")
    )

def ob_imbalance(df: pl.DataFrame):
    return df.with_columns(
        (pl.col("bid_size_1") / (pl.col("bid_size_1") + pl.col("ask_size_1"))).alias("imbalance")
    )

def mid_return(df: pl.DataFrame):
    assert("mid" in df.columns)
    return df.with_columns(
        pl.col("mid").pct_change().alias("ror_mid")
    )

def spread(df: pl.DataFrame):
    return df.with_columns(
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread")
    )

def cumulative_return(df: pl.DataFrame, window_size: int):
    return df.with_columns(
        pl.col("ror_mid").rolling_sum(window_size=window_size).alias("cum_return")
    )

def time_aggregation(df: pl.DataFrame) -> np.ndarray:
    ...

def bar_aggregation(df: pl.DataFrame, window_size: int = 20) -> np.ndarray:
    arr = df.to_numpy()
    sequence = [arr[i: i+window_size] for i in range(len(arr)-window_size)]
    return np.stack(sequence)

def get_lagged_cov(df: pl.DataFrame, c1: str, c2: str, window: int):

    return df.with_columns(
        pl.rolling_cov(pl.col(c1), pl.col(c2).shift(1), window_size=window).alias(f"cov_{c1}_{c2}")
    )