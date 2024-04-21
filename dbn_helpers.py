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

from features import *
from OrderBook import Book


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