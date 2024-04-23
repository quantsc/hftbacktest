from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import tqdm
from databento_dbn import FIXED_PRICE_SCALE, UNDEF_PRICE
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import *
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import polars as pl

def bid_ask_spread(df: pl.DataFrame) -> pl.Series:
    """A positive value indicating the current difference between the bid and ask prices in the current order books.
    See Kearns HFT page 7.
    """
    return df['ask_px_00'] - df['bid_px_00']

def mid_price(df: pl.DataFrame) -> pl.Series:
    """The average of the best bid and ask prices in the current order books."""
    return (df['ask_px_00'] + df['bid_px_00']) / 2

def weighted_mid_price(df: pl.DataFrame) -> pl.Series:
    """A variation on mid-price where the average of the bid and ask prices is weighted according to their inverse volume."""
    return (df['ask_px_00'] * df['bid_ct_00'] + df['bid_px_00'] * df['ask_ct_00']) / (df['bid_ct_00'] + df['ask_ct_00'])

def volume_imbalance(df: pl.DataFrame) -> pl.Series:
    """A signed quantity indicating the number of shares at the bid minus the number of shares at the ask in the current order books."""
    return df['bid_ct_00'] - df['ask_ct_00']

def log_return(df: pl.DataFrame) -> pl.Series:
    """The natural logarithm of the ratio of the current mid-price to the previous mid-price."""
    return (df['mid_price'] / df['mid_price'].shift(1)).ln()

# def rolling_signed_transaction_volume(df: pl.DataFrame, lookback: int=15) -> pl.Series:
#     """A signed quantity indicating the number of shares bought in the last lookback seconds minus the number of shares sold in the last lookback seconds."""
#     bought = df['purchase_ct'].rolling(lookback).sum()
#     sold = df['sale_ct'].rolling(lookback).sum()
#     return bought - sold

def rolling_weighted_mid_price(df: pl.DataFrame, lookback: int=15) -> pl.Series:
    """A variation on weighted mid-price where the average of the bid and ask prices is weighted according to their inverse volume in the last lookback seconds."""
    return (df['ask_px_00'] * df['bid_ct_00'] + df['bid_px_00'] * df['ask_ct_00']).rolling(lookback).sum() / (df['bid_ct_00'] + df['ask_ct_00']).rolling(lookback).sum()

def rolling_volatility(df: pl.DataFrame, lookback: int=15) -> pl.Series:
    """The standard deviation of log returns in the last lookback seconds."""
    return df['log_return'].rolling(lookback).std()

def rolling_volume_imbalance(df: pl.DataFrame, lookback: int=15) -> pl.Series:
    """A signed quantity indicating the number of shares at the bid minus the number of shares at the ask in the last lookback seconds."""
    return df['bid_ct_00'].rolling(lookback).sum() - df['ask_ct_00'].rolling(lookback).sum()

def target(df: pl.DataFrame, offset: int) -> pl.Series:
    """The natural logarithm of the ratio of the future mid-price to the current mid-price, offset by the given number of seconds."""
    return (df['mid_price'].shift(-offset) / df['mid_price']).ln()

def base_features(df: pl.DataFrame, aggregation: str = "1s") -> pl.DataFrame:
    """Build a DataFrame of base features from the given DataFrame.
    
    # Assumptions
    - The DataFrame has columns for the following:
    - ['ts_recv', 'action', 'side', 'price', 'size', 'symbol']

    # Returns
    - Number of each action [T = Trade, F = Fill, C = Cancel, M = Modify, A = Add]
    - Number of shares bought and sold
    - best bid and ask price
    - bid and ask size


    Notes: 
    - If many people are cancelling, or placing orders, that is information that can be used to predict future price movements.
    
    """
    return df.with_columns([
        bid_ask_spread(df),
        mid_price(df),
        weighted_mid_price(df),
        volume_imbalance(df),
    ])

def build_features(df: pl.DataFrame, lookback: int = 60, offset: int=5) -> pl.DataFrame:
    """Build a DataFrame of features from the given DataFrame.
    
    # Assumptions
    - The DataFrame has columns for the following:
        - `ask_px_00`: The best ask price in the current order books.
        - `bid_px_00`: The best bid price in the current order books.
        - `ask_ct_00`: The number of shares at the best ask price in the current order books.
        - `bid_ct_00`: The number of shares at the best bid price in the current order books.
        - `purchase_ct`: The number of shares bought in the last second.
        - `sale_ct`: The number of shares sold in the last second.
    # TODO: 
    - Support multiple depth levels 
    - Support multiple order types
    """
    feature_cols = [
        'log_return',
        'rolling_volume_imbalance',
        'rolling_weighted_mid_price',
        'rolling_volatility',
    ]
    df = df.with_columns(
        log_return=log_return(df),
        rolling_weighted_mid_price=rolling_weighted_mid_price(df, lookback),
        rolling_volatility=rolling_volatility(df, lookback),
        rolling_volume_imbalance=rolling_volume_imbalance(df, lookback),
    )
    for col in feature_cols:
        df = df.with_column(col + f"_mean_{lookback}", df[col].rolling(lookback).mean())
        df = df.with_column(col + f"_std_{lookback}", df[col].rolling(lookback).std())
        df = df.with_column(col + f"_sum_{lookback}", df[col].rolling(lookback).sum())
        df = df.with_column(col + f"_max_{lookback}", df[col].rolling(lookback).max())
        df = df.with_column(col + f"_min_{lookback}", df[col].rolling(lookback).min())
        df = df.with_column(col + f"_quantile_25_{lookback}", df[col].rolling(lookback).quantile(0.25))
        df = df.with_column(col + f"_quantile_75_{lookback}", df[col].rolling(lookback).quantile(0.75))
        df = df.with_column(col + f"_skew_{lookback}", df[col].rolling(lookback).skew())
        df = df.with_column(col + f"_kurtosis_{lookback}", df[col].rolling(lookback).kurtosis())
        df = df.with_column(col + f"_corr_{lookback}", df[col].rolling(lookback).corr())
    return df


def prepare_symbol(data: pl.DataFrame, date: str, symbol: str):
    # Filter by symbol
    df = data.clone()
    df = df.filter((df["symbol"] == symbol) & (df["size"] != 0))
    # Drop unnecessary columns
    df = df.drop(["__index_level_0__", "ts_recv", "channel_id", "publisher_id", "rtype", "instrument_id", "flags", "sequence", "ts_in_delta"])

    # 
    df = df.with_columns(df["ts_event"].cast(pl.Datetime))
    
    # Filter by action
    df = df.filter((pl.col("action") != "T") & (pl.col("action") != "F"))

    # Filter by date
    df = df.with_columns(df["ts_event"].cast(pl.Date).alias("is_date") == pl.lit(date).str.to_date()) 
    df = df.filter(df["is_date"] == True).drop(["is_date"])
    # Fix the datetimes                                
    df = df.with_columns(df["size"].cast(pl.Int16))

    return df

def get_data(symbol, start_date, end_date, base_path, remove_min = True):
    data = pl.read_parquet(f"{base_path}/mbp.parquet")
    df = prepare_symbol(data, start_date, end_date, symbol)
    if remove_min:
        df = df.with_columns(pl.col("ts_event") - pl.col("ts_event").min())
    return df


def build_book_from_mbo(df: pl.DataFrame) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    book = Book()
    best_bids_list = []
    best_asks_list = []
    num_rows = df.shape[0]
    for i, row in enumerate(tqdm.tqdm(df.iter_rows(named=True), total=num_rows)):
        best_bid, best_ask = book.bbo()
        best_bids_list.append({"ts_event": row["ts_event"], "price": best_bid.price, "size": best_bid.size, "total": best_bid.total_size})
        best_asks_list.append({"ts_event": row["ts_event"], "price": best_ask.price, "size": best_ask.size, "total": best_ask.total_size})
        book.apply(row)
    return best_bids_list, best_asks_list


def merge_bbo(best_bids_list, best_asks_list, unit="ms"):
    best_bids = pl.DataFrame(best_bids_list)
    best_asks = pl.DataFrame(best_asks_list)
    # Rename to best_bid_price and best_bid_size, best_ask_price and best_ask_size
    best_bids = best_bids.rename({"price": "bid_px_00", "size": "bid_ct_00", "total": "best_bid_total"})
    best_asks = best_asks.rename({"price": "ask_px_00", "size": "ask_ct_00", "total": "best_ask_total"})
    # divide by the fixed price scale
    best_bids = best_bids.with_columns([pl.col("bid_px_00") / FIXED_PRICE_SCALE])
    best_asks = best_asks.with_columns([pl.col("ask_px_00") / FIXED_PRICE_SCALE])
    # Forward fill the missing values

    if unit == "ms": 
        best_bids = best_bids.with_columns(pl.col("ts_event").dt.total_milliseconds())
        best_asks = best_asks.with_columns(pl.col("ts_event").dt.total_milliseconds())
        best_bids = best_bids.group_by("ts_event").agg(
            pl.col("bid_px_00").mean(), pl.col("bid_ct_00").sum(), pl.col("best_bid_total").last()
        )
        best_asks = best_asks.group_by("ts_event").agg(
            pl.col("ask_px_00").mean(), pl.col("ask_ct_00").sum(), pl.col("best_ask_total").last()
        )
        print(best_bids.shape, best_asks.shape)
    elif unit == "s":
        print(best_bids.shape, best_asks.shape)
        best_bids = best_bids.with_columns(pl.col("ts_event").dt.total_seconds())
        best_asks = best_asks.with_columns(pl.col("ts_event").dt.total_seconds())
        print(best_bids.shape, best_asks.shape)
        best_bids = best_bids.group_by("ts_event").agg(
            pl.col("bid_px_00").mean(), pl.col("bid_ct_00").sum(), pl.col("best_bid_total").last()
        )
        best_asks = best_asks.group_by("ts_event").agg(
            pl.col("ask_px_00").mean(), pl.col("ask_ct_00").sum(), pl.col("best_ask_total").last()
        )

    
    # TODO: Fix join type 
    merged = best_bids.join(best_asks, on="ts_event", how="inner")

    print(merged.shape, best_bids.shape, best_asks.shape)
    merged = merged.select(pl.all().forward_fill())
    return merged   


def prep_for_prediction(df: pl.DataFrame, offset=1000, lookback="1s"):
    df = df.with_columns(
        spread=bid_ask_spread(df),
        mid_price=mid_price(df),
        weighted_mid_price=weighted_mid_price(df),
        volume_imbalance=volume_imbalance(df)
    )
    feature_cols = [
        'volume_imbalance',
        'weighted_mid_price',
        'spread'
        # 'volatility', # TODO: Fix volatility calculation
    ]
    # convert from ms to datetime
    df = df.with_columns(pl.from_epoch("ts_event", time_unit="ms"))
    df = df.set_sorted("ts_event")
    for col in feature_cols:
        df=df.with_columns(pl.col(col).cast(pl.Float32))
    print(df.head)

    # TODO: Improve speed (cudf?)
    for col in tqdm.tqdm(feature_cols, total=len(feature_cols)):
        df = df.with_columns(pl.col(col).rolling_mean(window_size=lookback, by="ts_event").alias(f"rolling_{col}"))
        # df = df.with_columns(pl.col(col).rolling_min(window_size=lookback, by="ts_event").alias(f"rolling_{col}_min"))
        # df = df.with_columns(pl.col(col).rolling_max(window_size=lookback, by="ts_event").alias(f"rolling_{col}_max"))
        # df = df.with_columns(pl.col(col).rolling_std(window_size=lookback, by="ts_event").alias(f"rolling_{col}_std"))
        # df = df.with_columns(pl.col(col).rolling_sum(window_size=lookback, by="ts_event").alias(f"rolling_{col}_sum"))
        # df = df.with_columns(pl.col(col).rolling_median(window_size=lookback, by="ts_event").alias(f"rolling_{col}_median"))
        # df = df.with_columns(pl.col(col).rolling_skew(window_size=lookback, by="ts_event").alias(f"rolling_{col}_skew"))


    df = df.drop("ts_event")
    df = df.with_columns(pl.col("mid_price").diff().shift(-offset).alias("target"))[: -offset]
    # cast to f32
  
    return df


import databento as db
import os 
import pandas as pd
from databento_dbn import FIXED_PRICE_SCALE

current_dir = os.getcwd()

# Read all the files in the /data directory and store them in a list.
downloaded_files = [current_dir + '/fulldata/' + file for file in os.listdir(current_dir + '/fulldata') if file.endswith(".dbn.zst")]
save_dir = "/home/danny/hftbacktest/processed/"

# Make multithreaded
import multiprocessing
from joblib import Parallel, delayed

drop_cols = [
    "rtype",
    "publisher_id",
    "instrument_id",
    "flags",
    "sequence",
]

def process_file(file):
    # Read the data
    data = db.DBNStore.from_file(file).to_df(price_type="fixed")
    # Drop the columns that are not needed
    data["price"] = data["price"] / FIXED_PRICE_SCALE
    data = data.drop(columns=drop_cols).reset_index()
    # Save the data
    data.to_parquet(save_dir + file.split("/")[-1][:-8] + ".parquet")
    print(f"Processed {file}")

Parallel(n_jobs=multiprocessing.cpu_count() // 4)(delayed(process_file)(file) for file in sorted(downloaded_files))

