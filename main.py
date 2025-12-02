#!/usr/bin/env python3
"""
train_wandb.py

Self-contained training script for Mamba History Model with Weights & Biases integration.

Features:
- Full experiment tracking with W&B
- Hyperparameter tuning via W&B Sweeps (replaces Optuna)
- Model artifact versioning
- Comprehensive metrics logging
- Data visualization (tables, charts, confusion matrices)

Usage:
    # Single training run with defaults
    python train_wandb.py train --project mamba-history

    # Training with custom hyperparameters
    python train_wandb.py train --project mamba-history --d-model 384 --n-layers 5

    # Create a hyperparameter sweep
    python train_wandb.py sweep create --project mamba-history

    # Run sweep agent
    python train_wandb.py sweep run --sweep-id <sweep-id> --count 50
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:
    raise RuntimeError("wandb is required. Install with `uv add wandb` or `pip install wandb`.")

try:
    from mambapy.mamba import Mamba, MambaConfig
except Exception as e:
    raise RuntimeError(
        "mambapy is required. Install with `uv add mambapy` or `pip install mambapy`."
    ) from e


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Model architecture
    "d_model": 512,
    "n_layers": 6,
    "dropout_emb": 0.15,
    "dropout_mlp": 0.32,

    # Training
    "lr_peak": 0.0001,
    "steps": 5000,
    "warmup_steps": 400,
    "batch_size": 32,
    "seq_len": 256,
    "stride": 128,
    "weight_decay": 0.1,
    "grad_clip": 1.0,

    # Loss weights
    "time_nll": 0.72,
    "mark_scale": 0.85,

    # TPP head
    "mix_components": 3,
    "tau": 0.75,

    # Data
    "data_path": "data/raw/martin_lobster/RTSX_3days_2020-06-01_to_03_message_fixed.csv",
    "tick_size": 0.01,
    "dt_cap": 10.0,

    # Training control
    "seed": 1337,
    "patience": 10,
    "calibration_bins": 10,
    "eval_every": 100,

    # Features
    "use_time_of_day": True,
    "use_soft_binning": True,
}

# W&B Sweep configuration (Bayesian optimization)
SWEEP_CONFIG = {
    "name": "mamba-history-sweep",
    "method": "bayes",
    "metric": {
        "name": "val/mark_loss",
        "goal": "minimize"
    },
    "parameters": {
        # Architecture
        "d_model": {"values": [256, 320, 384, 512]},
        "n_layers": {"values": [4, 5, 6]},
        "dropout_emb": {"distribution": "uniform", "min": 0.10, "max": 0.25},
        "dropout_mlp": {"distribution": "uniform", "min": 0.20, "max": 0.40},

        # Optimizer
        "lr_peak": {"distribution": "log_uniform_values", "min": 5e-5, "max": 3e-4},
        "batch_size": {"values": [16, 24, 32, 48]},
        "seq_len": {"values": [128, 192, 256, 320]},

        # Loss weights
        "time_nll": {"distribution": "uniform", "min": 0.4, "max": 0.9},
        "mark_scale": {"distribution": "uniform", "min": 0.7, "max": 1.2},

        # TPP head
        "mix_components": {"values": [2, 3, 4]},
        "tau": {"values": [0.5, 0.75, 1.0, 1.25]},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 300,
        "eta": 2,
        "s": 3
    }
}

# Quick sweep for testing (grid search, fewer options)
SWEEP_CONFIG_QUICK = {
    "name": "mamba-history-sweep-quick",
    "method": "grid",
    "metric": {
        "name": "val/mark_loss",
        "goal": "minimize"
    },
    "parameters": {
        "d_model": {"values": [256, 384]},
        "n_layers": {"values": [4, 5]},
        "lr_peak": {"values": [0.0001, 0.00015]},
    }
}


# =============================================================================
# Utilities
# =============================================================================

SEED_DEFAULT = 1337


def set_seed(seed: int = SEED_DEFAULT) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer: Optional[str] = None) -> torch.device:
    prefer = (prefer or "").lower()

    def mps_available() -> bool:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise ValueError("CUDA requested but unavailable.")
    if prefer == "mps":
        if mps_available():
            return torch.device("mps")
        raise ValueError("MPS requested but unavailable.")
    if prefer == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Data Loading & Derived Features
# =============================================================================

@dataclass
class ExchangeConfig:
    data_path: str
    filename_template: str
    date_format: str
    column_mapping: Dict[str, str]
    time_unit: str = "s"  # s, ms, us, ns
    price_scale: float = 1.0


EXCHANGE_CONFIGS = {
    "mic": ExchangeConfig(
        data_path="data/mic/processed",
        filename_template="{date}__RTSX_FUT__{product}.parquet",
        date_format="%Y-%m-%d",
        column_mapping={
            "tst": "time",
            "event_code": "type",
            "vol": "size",
            "prc": "price",
            "is_buy": "direction",
            "order_id": "order_id",
        },
        time_unit="ns",
        price_scale=1.0,
    ),
    "lobster": ExchangeConfig(
        data_path="data/lobster/processed",
        filename_template="{product}_{date}_messages.parquet",
        date_format="%Y-%m-%d",
        column_mapping={
            "tst": "time",
            "event_code": "type",
            "vol": "size",
            "prc": "price",
            "direction": "direction",
            "order_id": "order_id",
        },
        time_unit="ns",
        price_scale=1.0,
    ),
}


def get_date_range(start: str, end: str) -> List[str]:
    """Generate a list of dates between start and end (inclusive)."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    delta = end_dt - start_dt
    return [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]


def load_messages(
    product: str,
    start_date: str,
    end_date: str,
    source: str,
    config: Optional[ExchangeConfig] = None,
) -> pl.DataFrame:
    """
    Unified dataloader for different sources.
    
    Args:
        source: Name of the source (key in EXCHANGE_CONFIGS).
        product: Product identifier (e.g. 'RTS2009', 'AAPL').
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        config: Optional custom configuration.
        
    Returns:
        pl.DataFrame with standard columns: time, type, size, price, direction.
    """
    cfg = config or EXCHANGE_CONFIGS.get(source)
    if cfg is None:
        raise ValueError(f"Unknown source: {source}")

    dates = get_date_range(start_date, end_date)
    dfs = []
    
    # Resolve base path relative to the script location or current working directory
    base_path = Path(cfg.data_path)
    
    for date_str in dates:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        fmt_date = dt.strftime(cfg.date_format)
        
        filename = cfg.filename_template.format(date=fmt_date, product=product)
        file_path = base_path / filename
        
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            # Read parquet using polars for speed and consistency with existing code
            df = pl.read_parquet(file_path)
            
            # Rename columns
            rename_map = {k: v for k, v in cfg.column_mapping.items() if k in df.columns}
            df = df.rename(rename_map)
            
            # Select required columns
            required_cols = ["time", "type", "size", "price", "direction"]
            # Check if all required columns are present
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"Warning: Missing columns {missing} in {file_path}")
                continue
                
            cols_to_select = required_cols + (["order_id"] if "order_id" in df.columns else [])
            df = df.select(cols_to_select)
            
            # Type conversions and scaling
            
            # Time
            if cfg.time_unit == "ns":
                df = df.with_columns(pl.col("time").cast(pl.Float64) / 1e9)
            elif cfg.time_unit == "ms":
                df = df.with_columns(pl.col("time").cast(pl.Float64) / 1e3)
            elif cfg.time_unit == "us":
                df = df.with_columns(pl.col("time").cast(pl.Float64) / 1e6)
            else:
                df = df.with_columns(pl.col("time").cast(pl.Float64))
                
            # Price
            if cfg.price_scale != 1.0:
                df = df.with_columns(pl.col("price") * cfg.price_scale)
            
            # Direction
            # Handle boolean direction if necessary (True/False -> 1/-1)
            if df.schema["direction"] == pl.Boolean:
                df = df.with_columns(
                    pl.when(pl.col("direction")).then(1).otherwise(-1).cast(pl.Int8).alias("direction")
                )
            
            # Cast other types to match train_wandb expectations
            df = df.with_columns([
                pl.col("type").cast(pl.Int32),
                pl.col("size").cast(pl.Float32),
                pl.col("price").cast(pl.Float32),
                pl.col("direction").cast(pl.Int8),
            ])
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not dfs:
        print(f"No data loaded for {exchange} {product} {start_date} to {end_date}")
        return pd.DataFrame(columns=["time", "type", "size", "price", "direction"])

    full_df = pl.concat(dfs)
    full_df = full_df.sort("time")
    
    return full_df

def load_lobster_messages(csv_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load a LOBSTER message file."""
    cols = ["time", "type", "order_id", "size", "price", "direction"]
    df = pd.read_csv(csv_path, header=None, names=cols, nrows=nrows)
    df["time"] = df["time"].astype(np.float64)
    df["type"] = df["type"].astype(np.int32)
    df["size"] = df["size"].astype(np.float32)
    df["price"] = (df["price"].astype(np.float32)) / 10000.0
    df["direction"] = df["direction"].astype(np.int8)
    return df


@dataclass
class DerivedFeatures:
    dp_ticks: np.ndarray
    log_size: np.ndarray
    dt_log: np.ndarray
    dt_prev: np.ndarray
    dt_next: np.ndarray
    has_next_event: np.ndarray
    type_code: np.ndarray
    side_code: np.ndarray
    level_proxy: np.ndarray
    tick_size: float
    time_scale: float
    time_absolute: np.ndarray


def compute_derived_features(
    df: pd.DataFrame,
    tick_size: float = 0.01,
    time_scale: Optional[float] = None,
) -> DerivedFeatures:
    """Compute relative features and inter-arrival statistics."""
    price = df["price"].to_numpy(dtype=np.float32)
    size = df["size"].to_numpy(dtype=np.float32)
    time_arr = df["time"].to_numpy(dtype=np.float64)
    msg_type = df["type"].to_numpy(dtype=np.int32)
    side_raw = df["direction"].to_numpy(dtype=np.int8)

    p_prev = np.roll(price, 1)
    p_prev[0] = price[0]
    dp = price - p_prev
    dp_ticks = np.rint(dp / float(tick_size)).astype(np.int32)

    dt_prev = np.diff(time_arr, prepend=time_arr[0])
    dt_prev = np.maximum(dt_prev, 0.0)
    if time_scale is None:
        positive = dt_prev[dt_prev > 0]
        if positive.size == 0:
            time_scale = 1e-3
        else:
            time_scale = float(max(np.median(positive), 1e-6))
    dt_log = np.log1p(dt_prev / float(time_scale)).astype(np.float32)

    log_size = np.log1p(size).astype(np.float32)

    dt_next = np.diff(time_arr, append=(time_arr[-1] + time_scale))
    has_next = np.ones_like(dt_next, dtype=bool)
    has_next[-1] = False

    abs_dp = np.abs(dp_ticks)
    level = np.zeros_like(abs_dp, dtype=np.int32)
    level[abs_dp == 0] = 0
    level[abs_dp >= 1] = 1

    return DerivedFeatures(
        dp_ticks=dp_ticks,
        log_size=log_size,
        dt_log=dt_log,
        dt_prev=dt_prev.astype(np.float32),
        dt_next=dt_next.astype(np.float32),
        has_next_event=has_next,
        type_code=msg_type,
        side_code=side_raw,
        level_proxy=level,
        tick_size=float(tick_size),
        time_scale=float(time_scale),
        time_absolute=time_arr.astype(np.float32),
    )


# =============================================================================
# Binning Utilities
# =============================================================================

@dataclass
class BinEdges:
    dp_ticks_center_range: Tuple[int, int] = (-2, 2)
    s_bins: int = 4
    t_bins: int = 4
    type_bins: int = 4
    side_bins: int = 2
    level_bins: int = 2


@dataclass
class FittedBins:
    dp_center_range: Tuple[int, int]
    log_size_edges: np.ndarray
    dt_log_edges: np.ndarray
    type_bins: int
    side_bins: int
    level_bins: int


@dataclass
class TokenBins:
    price_bins: np.ndarray
    size_bins: np.ndarray
    time_bins: np.ndarray
    type_bins: np.ndarray
    side_bins: np.ndarray
    level_bins: np.ndarray


def fit_quantile_edges(x: np.ndarray, num_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    q = np.linspace(0.0, 1.0, num_bins + 1)
    try:
        edges = np.quantile(x, q, method="linear")
    except TypeError:
        edges = np.quantile(x, q, interpolation="linear")
    edges = np.maximum.accumulate(edges)
    eps = 1e-9
    edges[0] -= eps
    edges[-1] += eps
    return edges.astype(np.float32)


def fit_bins(train_feats: DerivedFeatures, be: BinEdges) -> FittedBins:
    log_size_edges = fit_quantile_edges(train_feats.log_size, be.s_bins)
    dt_log_edges = fit_quantile_edges(train_feats.dt_log, be.t_bins)
    return FittedBins(
        dp_center_range=be.dp_ticks_center_range,
        log_size_edges=log_size_edges,
        dt_log_edges=dt_log_edges,
        type_bins=be.type_bins,
        side_bins=be.side_bins,
        level_bins=be.level_bins,
    )


def bin_dp_ticks(dp_ticks: np.ndarray, center_range: Tuple[int, int]) -> np.ndarray:
    lo, hi = int(center_range[0]), int(center_range[1])
    B_p = (hi - lo + 1) + 2
    out = np.empty_like(dp_ticks, dtype=np.int32)
    below = dp_ticks < lo
    above = dp_ticks > hi
    center = ~(below | above)
    out[below] = 0
    out[above] = B_p - 1
    out[center] = (dp_ticks[center] - lo + 1).astype(np.int32)
    return out


def bin_continuous_with_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(x.astype(np.float32), edges[1:-1], right=False)
    idx = np.clip(idx, 0, len(edges) - 2).astype(np.int32)
    return idx


def map_type_to_bin(msg_type: np.ndarray, B_ty: int) -> np.ndarray:
    mapping = {1: 0, 2: 1, 3: 2, 4: 3}
    out = np.full_like(msg_type, fill_value=B_ty - 1, dtype=np.int32)
    for k, v in mapping.items():
        out[msg_type == k] = v
    return np.clip(out, 0, B_ty - 1)


def map_side_to_bin(side: np.ndarray) -> np.ndarray:
    out = np.zeros_like(side, dtype=np.int32)
    out[side >= 0] = 1
    out[side < 0] = 0
    return out


def compute_token_bins(feats: DerivedFeatures, fb: FittedBins) -> TokenBins:
    p_bins = bin_dp_ticks(feats.dp_ticks, fb.dp_center_range)
    s_bins = bin_continuous_with_edges(feats.log_size, fb.log_size_edges)
    t_bins = bin_continuous_with_edges(feats.dt_log, fb.dt_log_edges)
    ty_bins = map_type_to_bin(feats.type_code, fb.type_bins)
    si_bins = map_side_to_bin(feats.side_code)
    lvl_bins = np.clip(feats.level_proxy, 0, fb.level_bins - 1)
    return TokenBins(
        price_bins=p_bins.astype(np.int32, copy=False),
        size_bins=s_bins.astype(np.int32, copy=False),
        time_bins=t_bins.astype(np.int32, copy=False),
        type_bins=ty_bins.astype(np.int32, copy=False),
        side_bins=si_bins.astype(np.int32, copy=False),
        level_bins=lvl_bins.astype(np.int32, copy=False),
    )


# =============================================================================
# Hierarchical Grouping
# =============================================================================

def _normalize_coarse_count(num_fine: int, desired: Optional[int]) -> int:
    if desired is None or desired <= 0:
        return num_fine
    return int(max(1, min(num_fine, desired)))


def _linear_neighbors(num_items: int) -> List[List[int]]:
    neigh: List[List[int]] = []
    for i in range(num_items):
        cur: List[int] = []
        if i - 1 >= 0:
            cur.append(i - 1)
        if i + 1 < num_items:
            cur.append(i + 1)
        neigh.append(cur)
    return neigh


def build_equal_groups(num_fine: int, desired_coarse: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    num_coarse = _normalize_coarse_count(num_fine, desired_coarse)
    if num_coarse == 1:
        fine_to_coarse = np.zeros(num_fine, dtype=np.int64)
        fine_to_resid = np.arange(num_fine, dtype=np.int64)
        group_sizes = np.array([num_fine], dtype=np.int64)
        return fine_to_coarse, fine_to_resid, group_sizes, 1
    splits = np.array_split(np.arange(num_fine, dtype=np.int64), num_coarse)
    splits = [s for s in splits if len(s) > 0]
    num_coarse = len(splits)
    fine_to_coarse = np.zeros(num_fine, dtype=np.int64)
    fine_to_resid = np.zeros(num_fine, dtype=np.int64)
    group_sizes = np.zeros(num_coarse, dtype=np.int64)
    for c, idxs in enumerate(splits):
        group_sizes[c] = len(idxs)
        fine_to_coarse[idxs] = c
        fine_to_resid[idxs] = np.arange(len(idxs), dtype=np.int64)
    return fine_to_coarse, fine_to_resid, group_sizes, num_coarse


def build_price_groups(
    num_fine: int,
    dp_center_range: Tuple[int, int],
    desired_coarse: Optional[int],
    scheme: str = "sign5",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    scheme = (scheme or "sign5").lower()
    lo, hi = int(dp_center_range[0]), int(dp_center_range[1])
    center_span = hi - lo + 1
    tails = num_fine - center_span
    assert tails in (1, 2), "Expected two price tails."
    neg_tail = 0
    pos_tail = num_fine - 1

    fine_to_coarse = np.zeros(num_fine, dtype=np.int64)
    fine_to_resid = np.zeros(num_fine, dtype=np.int64)

    if scheme == "sign3":
        neg_group: List[int] = []
        zero_group: List[int] = []
        pos_group: List[int] = []
        for idx in range(num_fine):
            if idx == neg_tail:
                neg_group.append(idx)
                continue
            if idx == pos_tail:
                pos_group.append(idx)
                continue
            rel = idx - 1
            dp_val = lo + rel
            if dp_val < 0:
                neg_group.append(idx)
            elif dp_val > 0:
                pos_group.append(idx)
            else:
                zero_group.append(idx)
        coarse_splits = [
            np.array(neg_group, dtype=np.int64),
            np.array(zero_group, dtype=np.int64),
            np.array(pos_group, dtype=np.int64),
        ]
    else:
        neg_far: List[int] = []
        neg_near: List[int] = []
        zero_group: List[int] = []
        pos_near: List[int] = []
        pos_far: List[int] = []
        threshold = 1
        for idx in range(num_fine):
            if idx == neg_tail:
                neg_far.append(idx)
                continue
            if idx == pos_tail:
                pos_far.append(idx)
                continue
            rel = idx - 1
            dp_val = lo + rel
            if dp_val < 0:
                if abs(dp_val) <= threshold:
                    neg_near.append(idx)
                else:
                    neg_far.append(idx)
            elif dp_val > 0:
                if abs(dp_val) <= threshold:
                    pos_near.append(idx)
                else:
                    pos_far.append(idx)
            else:
                zero_group.append(idx)
        coarse_splits = [
            np.array(neg_far, dtype=np.int64),
            np.array(neg_near, dtype=np.int64),
            np.array(zero_group, dtype=np.int64),
            np.array(pos_near, dtype=np.int64),
            np.array(pos_far, dtype=np.int64),
        ]
        coarse_splits = [grp for grp in coarse_splits if len(grp) > 0]

    num_coarse = len(coarse_splits)
    desired = _normalize_coarse_count(num_coarse, desired_coarse)
    if desired < num_coarse:
        coarse_splits = np.array_split(np.concatenate(coarse_splits), desired)
        num_coarse = desired

    group_sizes = np.zeros(num_coarse, dtype=np.int64)
    for c, idxs in enumerate(coarse_splits):
        group_sizes[c] = len(idxs)
        fine_to_coarse[idxs] = c
        fine_to_resid[idxs] = np.arange(len(idxs), dtype=np.int64)

    return fine_to_coarse, fine_to_resid, group_sizes, num_coarse


@dataclass
class HierFieldMap:
    name: str
    num_fine: int
    num_coarse: int
    max_resid: int
    fine_to_coarse: np.ndarray
    fine_to_resid: np.ndarray
    group_sizes: np.ndarray
    coarse_neighbors: List[List[int]]
    resid_neighbors: List[List[List[int]]]
    coarse_resid_to_fine: np.ndarray

    def valid_mask(self) -> np.ndarray:
        mask = np.zeros((self.num_coarse, self.max_resid), dtype=bool)
        for c in range(self.num_coarse):
            width = int(self.group_sizes[c])
            if width <= 0:
                continue
            mask[c, :width] = True
        return mask


def _build_resid_neighbors(group_sizes: np.ndarray, max_resid: int) -> List[List[List[int]]]:
    neighbors: List[List[List[int]]] = []
    for c, size in enumerate(group_sizes):
        group: List[List[int]] = []
        width = int(size)
        for r in range(max_resid):
            if r >= width:
                group.append([])
                continue
            cur: List[int] = []
            if r - 1 >= 0:
                cur.append(r - 1)
            if r + 1 < width:
                cur.append(r + 1)
            group.append(cur)
        neighbors.append(group)
    return neighbors


@dataclass
class HierConfig:
    price_coarse: int = 5
    price_scheme: str = "sign5"
    size_coarse: int = 4
    time_coarse: int = 4
    type_coarse: Optional[int] = None
    side_coarse: Optional[int] = None
    level_coarse: Optional[int] = None


@dataclass
class HierMaps:
    price: HierFieldMap
    size: HierFieldMap
    time: HierFieldMap
    type: HierFieldMap
    side: HierFieldMap
    level: HierFieldMap


def build_hier_field_map(
    name: str,
    num_fine: int,
    desired_coarse: Optional[int],
    scheme: Optional[str] = None,
    dp_center_range: Optional[Tuple[int, int]] = None,
) -> HierFieldMap:
    if name == "price":
        if dp_center_range is None:
            raise ValueError("dp_center_range required for price map.")
        fine_to_coarse, fine_to_resid, group_sizes, num_coarse = build_price_groups(
            num_fine=num_fine,
            dp_center_range=dp_center_range,
            desired_coarse=desired_coarse,
            scheme=scheme or "sign5",
        )
    else:
        fine_to_coarse, fine_to_resid, group_sizes, num_coarse = build_equal_groups(num_fine, desired_coarse)
    max_resid = int(group_sizes.max()) if group_sizes.size > 0 else 1
    coarse_resid_to_fine = np.full((num_coarse, max_resid), -1, dtype=np.int64)
    for fine_idx in range(num_fine):
        c = int(fine_to_coarse[fine_idx])
        r = int(fine_to_resid[fine_idx])
        if 0 <= c < num_coarse and 0 <= r < max_resid:
            coarse_resid_to_fine[c, r] = fine_idx
    return HierFieldMap(
        name=name,
        num_fine=num_fine,
        num_coarse=num_coarse,
        max_resid=max_resid,
        fine_to_coarse=fine_to_coarse.astype(np.int64),
        fine_to_resid=fine_to_resid.astype(np.int64),
        group_sizes=group_sizes.astype(np.int64),
        coarse_neighbors=_linear_neighbors(num_coarse),
        resid_neighbors=_build_resid_neighbors(group_sizes, max_resid),
        coarse_resid_to_fine=coarse_resid_to_fine,
    )


@dataclass
class InputSpec:
    B_p: int
    B_s: int
    B_t: int
    B_ty: int
    B_si: int
    B_l: int


def build_hier_maps(spec: InputSpec, cfg: HierConfig, dp_center_range: Tuple[int, int]) -> HierMaps:
    return HierMaps(
        price=build_hier_field_map("price", spec.B_p, cfg.price_coarse, cfg.price_scheme, dp_center_range),
        size=build_hier_field_map("size", spec.B_s, cfg.size_coarse),
        time=build_hier_field_map("time", spec.B_t, cfg.time_coarse),
        type=build_hier_field_map("type", spec.B_ty, cfg.type_coarse),
        side=build_hier_field_map("side", spec.B_si, cfg.side_coarse),
        level=build_hier_field_map("level", spec.B_l, cfg.level_coarse),
    )


# =============================================================================
# Targets
# =============================================================================

@dataclass
class NextEventTargets:
    price_coarse: np.ndarray
    price_resid: np.ndarray
    size_coarse: np.ndarray
    size_resid: np.ndarray
    time_coarse: np.ndarray
    time_resid: np.ndarray
    type_coarse: np.ndarray
    type_resid: np.ndarray
    side_coarse: np.ndarray
    side_resid: np.ndarray
    level_coarse: np.ndarray
    level_resid: np.ndarray
    has_event: np.ndarray


def make_next_event_targets(token_bins: TokenBins, maps: HierMaps) -> NextEventTargets:
    N = len(token_bins.price_bins)
    valid = np.zeros(N, dtype=bool)
    valid[:-1] = True

    def lift(field_map: HierFieldMap, fine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        coarse = field_map.fine_to_coarse[fine]
        resid = field_map.fine_to_resid[fine]
        return coarse, resid

    def shift(arr: np.ndarray) -> np.ndarray:
        out = np.empty_like(arr)
        out[:-1] = arr[1:]
        out[-1] = 0
        return out

    price_coarse, price_resid = lift(maps.price, shift(token_bins.price_bins))
    size_coarse, size_resid = lift(maps.size, shift(token_bins.size_bins))
    time_coarse, time_resid = lift(maps.time, shift(token_bins.time_bins))
    type_coarse, type_resid = lift(maps.type, shift(token_bins.type_bins))
    side_coarse, side_resid = lift(maps.side, shift(token_bins.side_bins))
    level_coarse, level_resid = lift(maps.level, shift(token_bins.level_bins))

    return NextEventTargets(
        price_coarse=price_coarse.astype(np.int64, copy=False),
        price_resid=price_resid.astype(np.int64, copy=False),
        size_coarse=size_coarse.astype(np.int64, copy=False),
        size_resid=size_resid.astype(np.int64, copy=False),
        time_coarse=time_coarse.astype(np.int64, copy=False),
        time_resid=time_resid.astype(np.int64, copy=False),
        type_coarse=type_coarse.astype(np.int64, copy=False),
        type_resid=type_resid.astype(np.int64, copy=False),
        side_coarse=side_coarse.astype(np.int64, copy=False),
        side_resid=side_resid.astype(np.int64, copy=False),
        level_coarse=level_coarse.astype(np.int64, copy=False),
        level_resid=level_resid.astype(np.int64, copy=False),
        has_event=valid,
    )


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class SequenceConfig:
    seq_len: int = 256
    stride: int = 128
    add_bos: bool = False


@dataclass
class LossWeights:
    time_nll: float = 0.7
    price_coarse: float = 0.8
    price_resid: float = 0.5
    size_coarse: float = 0.6
    size_resid: float = 0.4
    time_coarse: float = 0.5
    time_resid: float = 0.4
    type_coarse: float = 0.4
    type_resid: float = 0.15
    side_coarse: float = 0.25
    side_resid: float = 0.0
    level_coarse: float = 0.25
    level_resid: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


@dataclass
class SmoothingConfig:
    price_coarse: float = 0.02
    price_resid: float = 0.01
    size_coarse: float = 0.02
    size_resid: float = 0.01
    time_coarse: float = 0.02
    time_resid: float = 0.01
    type_coarse: float = 0.01
    type_resid: float = 0.005
    side_coarse: float = 0.0
    side_resid: float = 0.0
    level_coarse: float = 0.0
    level_resid: float = 0.0


# =============================================================================
# Dataset
# =============================================================================

class HistoryDataset(Dataset):
    def __init__(
        self,
        token_bins: TokenBins,
        targets: NextEventTargets,
        feats: DerivedFeatures,
        seq_cfg: SequenceConfig,
        dt_cap: float,
        tau_seconds: float,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.seq = seq_cfg
        self.ignore_index = int(ignore_index)
        self.dt_cap = float(max(dt_cap, tau_seconds))
        self.tau = float(tau_seconds)

        self.type_bins = token_bins.type_bins
        self.side_bins = token_bins.side_bins
        self.price_bins = token_bins.price_bins
        self.size_bins = token_bins.size_bins
        self.time_bins = token_bins.time_bins
        self.level_bins = token_bins.level_bins

        self.targets = targets
        self.dt_next = np.minimum(feats.dt_next, self.dt_cap).astype(np.float32, copy=False)
        censor = feats.dt_next > self.dt_cap
        censor |= ~targets.has_event
        self.censor = censor.astype(np.bool_, copy=False)

        dt_prev = np.minimum(feats.dt_prev, self.dt_cap)
        self.dt_prev = dt_prev.astype(np.float32, copy=False)

        self.time_absolute = feats.time_absolute.astype(np.float32, copy=False)
        self.log_size_continuous = feats.log_size.astype(np.float32, copy=False)
        self.dt_log_continuous = feats.dt_log.astype(np.float32, copy=False)

        L = self.seq.seq_len
        stride = max(1, int(self.seq.stride))
        last_start = len(self.type_bins) - L
        if last_start < 0:
            self.starts = np.array([], dtype=np.int64)
        else:
            self.starts = np.arange(0, last_start + 1, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = int(self.starts[idx])
        e = s + self.seq.seq_len

        inputs = {
            "msg_type": torch.from_numpy(self.type_bins[s:e].astype(np.int64)),
            "side": torch.from_numpy(self.side_bins[s:e].astype(np.int64)),
            "price": torch.from_numpy(self.price_bins[s:e].astype(np.int64)),
            "size": torch.from_numpy(self.size_bins[s:e].astype(np.int64)),
            "time": torch.from_numpy(self.time_bins[s:e].astype(np.int64)),
            "level": torch.from_numpy(self.level_bins[s:e].astype(np.int64)),
        }

        dt_prev = torch.from_numpy(self.dt_prev[s:e].astype(np.float32))
        dt_next = torch.from_numpy(self.dt_next[s:e].astype(np.float32))
        censor = torch.from_numpy(self.censor[s:e].astype(np.bool_))
        time_abs = torch.from_numpy(self.time_absolute[s:e].astype(np.float32))
        log_size_cont = torch.from_numpy(self.log_size_continuous[s:e].astype(np.float32))
        dt_log_cont = torch.from_numpy(self.dt_log_continuous[s:e].astype(np.float32))

        def slice_with_mask(arr: np.ndarray) -> torch.Tensor:
            seg = arr[s:e].astype(np.int64)
            seg = np.where(censor.numpy(), self.ignore_index, seg)
            seg[-1] = self.ignore_index
            return torch.from_numpy(seg)

        targets = {
            "price_coarse": slice_with_mask(self.targets.price_coarse),
            "price_resid": slice_with_mask(self.targets.price_resid),
            "size_coarse": slice_with_mask(self.targets.size_coarse),
            "size_resid": slice_with_mask(self.targets.size_resid),
            "time_coarse": slice_with_mask(self.targets.time_coarse),
            "time_resid": slice_with_mask(self.targets.time_resid),
            "type_coarse": slice_with_mask(self.targets.type_coarse),
            "type_resid": slice_with_mask(self.targets.type_resid),
            "side_coarse": slice_with_mask(self.targets.side_coarse),
            "side_resid": slice_with_mask(self.targets.side_resid),
            "level_coarse": slice_with_mask(self.targets.level_coarse),
            "level_resid": slice_with_mask(self.targets.level_resid),
        }

        attn = torch.ones(self.seq.seq_len, dtype=torch.bool)
        return {
            "inputs": inputs,
            "dt_prev": dt_prev,
            "dt_next": dt_next,
            "censor": censor,
            "targets": targets,
            "attention_mask": attn,
            "time_absolute": time_abs,
            "log_size_continuous": log_size_cont,
            "dt_log_continuous": dt_log_cont,
        }


def collate_history(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    inputs = {}
    for key in batch[0]["inputs"]:
        inputs[key] = torch.stack([b["inputs"][key] for b in batch], dim=0)
    dt_prev = torch.stack([b["dt_prev"] for b in batch], dim=0)
    dt_next = torch.stack([b["dt_next"] for b in batch], dim=0)
    censor = torch.stack([b["censor"] for b in batch], dim=0)
    attn = torch.stack([b["attention_mask"] for b in batch], dim=0)
    time_abs = torch.stack([b["time_absolute"] for b in batch], dim=0)
    log_size_cont = torch.stack([b["log_size_continuous"] for b in batch], dim=0)
    dt_log_cont = torch.stack([b["dt_log_continuous"] for b in batch], dim=0)
    targets = {}
    for key in batch[0]["targets"]:
        targets[key] = torch.stack([b["targets"][key] for b in batch], dim=0)
    return {
        "inputs": inputs,
        "dt_prev": dt_prev,
        "dt_next": dt_next,
        "censor": censor,
        "targets": targets,
        "attention_mask": attn,
        "time_absolute": time_abs,
        "log_size_continuous": log_size_cont,
        "dt_log_continuous": dt_log_cont,
    }


# =============================================================================
# Model Components
# =============================================================================

class MixtureTPPHead(nn.Module):
    """Mixture-of-exponentials conditional intensity."""

    def __init__(
        self,
        d_model: int,
        num_components: int = 3,
        min_beta: float = 1e-4,
        temperature_init: float = 0.0,
    ):
        super().__init__()
        self.num_components = int(max(1, num_components))
        self.alpha_proj = nn.Linear(d_model, self.num_components)
        self.beta_proj = nn.Linear(d_model, self.num_components)
        self.register_parameter("log_temperature", nn.Parameter(torch.tensor(float(temperature_init))))
        self.min_beta = float(min_beta)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_alpha = self.alpha_proj(hidden)
        raw_beta = self.beta_proj(hidden)
        alpha = F.softplus(raw_alpha) * torch.exp(self.log_temperature)
        beta = F.softplus(raw_beta) + self.min_beta
        return alpha, beta

    @staticmethod
    def log_stats(alpha: torch.Tensor, beta: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = dt.unsqueeze(-1)
        exp_term = torch.exp(-beta * dt)
        intensity = (alpha * exp_term).sum(dim=-1).clamp_min(1e-12)
        integral = (alpha / beta) * (1.0 - exp_term)
        Lambda = integral.sum(dim=-1).clamp_min(1e-12)
        log_intensity = torch.log(intensity)
        return log_intensity, Lambda

    @staticmethod
    def prob_within(alpha: torch.Tensor, beta: torch.Tensor, tau: float) -> torch.Tensor:
        dt = torch.as_tensor(tau, dtype=alpha.dtype, device=alpha.device)
        exp_term = torch.exp(-beta * dt)
        integral = (alpha / beta) * (1.0 - exp_term)
        Lambda = integral.sum(dim=-1)
        return (1.0 - torch.exp(-Lambda)).clamp(0.0, 1.0)


class HierHead(nn.Module):
    def __init__(self, d_model: int, field_map: HierFieldMap, context_dim: Optional[int] = None, context_dropout: float = 0.2):
        super().__init__()
        self.field_map = field_map
        self.context_dim = int(context_dim or max(8, d_model // 4))
        self.has_resid = field_map.max_resid > 1
        self.coarse = nn.Linear(d_model, field_map.num_coarse)
        self.logit_temperature = nn.Parameter(torch.zeros(1))
        if self.has_resid:
            self.coarse_embed = nn.Parameter(torch.randn(field_map.num_coarse, self.context_dim))
            nn.init.normal_(self.coarse_embed, mean=0.0, std=1.0 / math.sqrt(field_map.num_coarse))
            self.resid = nn.Linear(d_model + self.context_dim, field_map.num_coarse * field_map.max_resid)
            self.ctx_drop = nn.Dropout(p=float(max(0.0, min(1.0, context_dropout))))
        else:
            self.register_parameter("coarse_embed", None)
            self.resid = None
            self.ctx_drop = None

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        coarse_logits = self.coarse(hidden) * torch.exp(self.logit_temperature)
        out: Dict[str, torch.Tensor] = {"coarse": coarse_logits}
        if self.has_resid:
            probs = torch.softmax(coarse_logits.detach(), dim=-1)
            ctx = torch.matmul(probs, self.coarse_embed)
            if self.ctx_drop is not None:
                ctx = self.ctx_drop(ctx)
            resid_input = torch.cat([hidden, ctx], dim=-1)
            resid_logits = self.resid(resid_input)
            B, L, _ = resid_logits.shape
            resid_logits = resid_logits.view(B, L, self.field_map.num_coarse, self.field_map.max_resid)
            out["resid"] = resid_logits
        else:
            out["resid"] = None
        return out


class CyclicalTimeEncoding(nn.Module):
    """Encode absolute time-of-day using cyclical (sin/cos) features."""

    def __init__(self, d_model: int):
        super().__init__()
        self.time_proj = nn.Linear(4, d_model)

    def encode_time(self, seconds_after_midnight: torch.Tensor) -> torch.Tensor:
        hours = (seconds_after_midnight / 3600.0) % 24.0
        minutes = (seconds_after_midnight / 60.0) % 60.0
        hour_sin = torch.sin(2 * math.pi * hours / 24.0)
        hour_cos = torch.cos(2 * math.pi * hours / 24.0)
        minute_sin = torch.sin(2 * math.pi * minutes / 60.0)
        minute_cos = torch.cos(2 * math.pi * minutes / 60.0)
        return torch.stack([hour_sin, hour_cos, minute_sin, minute_cos], dim=-1)

    def forward(self, seconds_after_midnight: torch.Tensor) -> torch.Tensor:
        cyclical_feats = self.encode_time(seconds_after_midnight)
        return self.time_proj(cyclical_feats)


class SoftBinning(nn.Module):
    """Soft (fuzzy) binning for continuous features."""

    def __init__(self, num_bins: int, embed_dim: int, temperature: float = 1.0):
        super().__init__()
        self.num_bins = num_bins
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(num_bins, embed_dim)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(self, x: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        distances = torch.abs(x.unsqueeze(-1) - bin_centers.unsqueeze(0).unsqueeze(0))
        temperature = torch.exp(self.log_temperature)
        weights = torch.softmax(-distances / temperature, dim=-1)
        all_embeds = self.embeddings.weight
        weighted_emb = torch.matmul(weights, all_embeds)
        return weighted_emb


class HistoryModel(nn.Module):
    def __init__(
        self,
        spec: InputSpec,
        hier_maps: HierMaps,
        d_model: int = 256,
        n_layers: int = 4,
        dropout_emb: float = 0.1,
        dropout_mlp: float = 0.1,
        mix_components: int = 3,
        use_soft_binning: bool = False,
        fitted_bins: Optional[FittedBins] = None,
    ):
        super().__init__()
        self.spec = spec
        self.hier_maps = hier_maps
        self.use_soft_binning = use_soft_binning
        self.field_info: List[Tuple[str, int]] = [
            ("msg_type", spec.B_ty),
            ("side", spec.B_si),
            ("price", spec.B_p),
            ("size", spec.B_s),
            ("time", spec.B_t),
            ("level", spec.B_l),
        ]
        self.field_names = [name for name, _ in self.field_info]
        embed_dim = max(16, min(64, d_model // 4))
        self.embed_dim = embed_dim

        self.embeddings = nn.ModuleDict()
        for name, count in self.field_info:
            if use_soft_binning and name in ["size", "time"]:
                self.embeddings[name] = SoftBinning(count, embed_dim, temperature=1.0)
            else:
                self.embeddings[name] = nn.Embedding(count, embed_dim)

        if use_soft_binning and fitted_bins is not None:
            self.register_buffer('log_size_edges', torch.from_numpy(fitted_bins.log_size_edges))
            self.register_buffer('dt_log_edges', torch.from_numpy(fitted_bins.dt_log_edges))
        else:
            self.log_size_edges = None
            self.dt_log_edges = None

        proj_in = embed_dim * len(self.field_info)
        self.proj = nn.Linear(proj_in, d_model)
        self.dt_proj = nn.Linear(1, d_model)
        self.time_of_day_encoder = CyclicalTimeEncoding(d_model)
        self.emb_drop = nn.Dropout(p=float(dropout_emb))
        cfg = MambaConfig(d_model=d_model, n_layers=n_layers)
        self.mamba = Mamba(cfg)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p=float(dropout_mlp))
        context_dim = max(8, d_model // 4)
        raw_heads = {
            "price": HierHead(d_model, hier_maps.price, context_dim=context_dim),
            "size": HierHead(d_model, hier_maps.size, context_dim=context_dim),
            "time": HierHead(d_model, hier_maps.time, context_dim=context_dim),
            "type": HierHead(d_model, hier_maps.type, context_dim=context_dim),
            "side": HierHead(d_model, hier_maps.side, context_dim=context_dim),
            "level": HierHead(d_model, hier_maps.level, context_dim=context_dim),
        }
        self.head_keys = {field: (field if field != "type" else "msg_type_head") for field in raw_heads}
        self.heads = nn.ModuleDict({self.head_keys[field]: head for field, head in raw_heads.items()})
        self.tpp_head = MixtureTPPHead(d_model, num_components=mix_components)

    def encode(self, inputs: Dict[str, torch.Tensor], dt_prev: torch.Tensor,
               time_absolute: Optional[torch.Tensor] = None,
               log_size_continuous: Optional[torch.Tensor] = None,
               dt_log_continuous: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeds = []
        for name in self.field_names:
            if self.use_soft_binning and name == "size" and log_size_continuous is not None:
                embeds.append(self.embeddings[name](log_size_continuous, self.log_size_edges))
            elif self.use_soft_binning and name == "time" and dt_log_continuous is not None:
                embeds.append(self.embeddings[name](dt_log_continuous, self.dt_log_edges))
            else:
                embeds.append(self.embeddings[name](inputs[name]))

        emb = torch.cat(embeds, dim=-1)
        x = self.proj(emb)

        dt_feat = torch.log1p(dt_prev).unsqueeze(-1)
        x = x + self.dt_proj(dt_feat)

        if time_absolute is not None:
            time_emb = self.time_of_day_encoder(time_absolute)
            x = x + time_emb

        x = self.emb_drop(x)
        x = self.mamba(x)
        x = self.norm(x)
        x = self.drop(x)
        return x

    def forward(self, inputs: Dict[str, torch.Tensor], dt_prev: torch.Tensor,
                time_absolute: Optional[torch.Tensor] = None,
                log_size_continuous: Optional[torch.Tensor] = None,
                dt_log_continuous: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        hidden = self.encode(inputs, dt_prev, time_absolute, log_size_continuous, dt_log_continuous)
        alpha, beta = self.tpp_head(hidden)
        mark_logits = {field: self.heads[key](hidden) for field, key in self.head_keys.items()}
        return (alpha, beta), mark_logits


# =============================================================================
# Loss Helpers
# =============================================================================

@dataclass
class FieldLossHelper:
    map: HierFieldMap
    valid_mask: torch.Tensor
    coarse_smoothing: torch.Tensor
    resid_smoothing: torch.Tensor

    @property
    def has_resid(self) -> bool:
        return self.map.max_resid > 1


def _build_coarse_smoothing(field_map: HierFieldMap, smoothing: float) -> np.ndarray:
    num = field_map.num_coarse
    table = np.zeros((num, num), dtype=np.float32)
    smoothing = float(max(0.0, min(0.99, smoothing)))
    for idx in range(num):
        neighbors = field_map.coarse_neighbors[idx]
        if smoothing > 0.0 and neighbors:
            share = smoothing / len(neighbors)
            table[idx, idx] = 1.0 - smoothing
            for n in neighbors:
                table[idx, n] += share
        else:
            table[idx, idx] = 1.0
    return table


def _build_resid_smoothing(field_map: HierFieldMap, smoothing: float) -> np.ndarray:
    num_c = field_map.num_coarse
    max_r = field_map.max_resid
    table = np.zeros((num_c, max_r, max_r), dtype=np.float32)
    mask = field_map.valid_mask()
    smoothing = float(max(0.0, min(0.99, smoothing)))
    for c in range(num_c):
        width = int(field_map.group_sizes[c])
        for r in range(max_r):
            if not mask[c, r]:
                continue
            neighbors = field_map.resid_neighbors[c][r]
            if smoothing > 0.0 and neighbors:
                share = smoothing / len(neighbors)
                table[c, r, r] = 1.0 - smoothing
                for n in neighbors:
                    table[c, r, n] += share
            else:
                table[c, r, r] = 1.0
    return table


def build_loss_helpers(hier_maps: HierMaps, smoothing: SmoothingConfig, device: torch.device) -> Dict[str, FieldLossHelper]:
    helpers: Dict[str, FieldLossHelper] = {}

    def make(name: str, field_map: HierFieldMap, smooth_coarse: float, smooth_resid: float):
        helpers[name] = FieldLossHelper(
            map=field_map,
            valid_mask=torch.from_numpy(field_map.valid_mask()).to(device=device),
            coarse_smoothing=torch.from_numpy(_build_coarse_smoothing(field_map, smooth_coarse)).to(device=device),
            resid_smoothing=torch.from_numpy(_build_resid_smoothing(field_map, smooth_resid)).to(device=device),
        )

    make("price", hier_maps.price, smoothing.price_coarse, smoothing.price_resid)
    make("size", hier_maps.size, smoothing.size_coarse, smoothing.size_resid)
    make("time", hier_maps.time, smoothing.time_coarse, smoothing.time_resid)
    make("type", hier_maps.type, smoothing.type_coarse, smoothing.type_resid)
    make("side", hier_maps.side, smoothing.side_coarse, smoothing.side_resid)
    make("level", hier_maps.level, smoothing.level_coarse, smoothing.level_resid)
    return helpers


def _semantic_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing_table: torch.Tensor,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = logits.new_zeros(())
    zero_long = logits.new_zeros((), dtype=torch.long)
    flat_tgt = targets.view(-1)
    valid = flat_tgt != ignore_index
    if not torch.any(valid):
        return zero, zero_long, zero_long
    logits_flat = logits.view(-1, logits.size(-1))[valid]
    tgt = flat_tgt[valid].long()
    log_probs = F.log_softmax(logits_flat, dim=-1)
    target_dist = smoothing_table.index_select(0, tgt)
    loss = -(target_dist * log_probs).sum(dim=-1).mean()
    pred = logits_flat.argmax(dim=-1)
    correct = (pred == tgt).sum()
    count = torch.tensor(tgt.numel(), device=logits.device, dtype=torch.long)
    return loss, correct, count


def _semantic_ce_resid(
    logits: torch.Tensor,
    coarse_targets: torch.Tensor,
    resid_targets: torch.Tensor,
    helper: FieldLossHelper,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = logits.new_zeros(())
    zero_long = logits.new_zeros((), dtype=torch.long)
    coarse_flat = coarse_targets.view(-1)
    resid_flat = resid_targets.view(-1)
    valid = (coarse_flat != ignore_index) & (resid_flat != ignore_index)
    if not torch.any(valid):
        return zero, zero_long, zero_long
    logits_flat = logits.view(-1, logits.size(2), logits.size(3))[valid]
    coarse_idx = coarse_flat[valid].long()
    resid_idx = resid_flat[valid].long()
    gather = logits_flat[torch.arange(len(logits_flat), device=logits.device), coarse_idx]
    valid_mask = helper.valid_mask.index_select(0, coarse_idx)
    min_value = torch.finfo(gather.dtype).min
    gather = torch.where(valid_mask, gather, min_value)
    log_probs = F.log_softmax(gather, dim=-1)
    smoothing = helper.resid_smoothing.index_select(0, coarse_idx)
    smoothing = smoothing[torch.arange(len(logits_flat), device=logits.device), resid_idx]
    loss = -(smoothing * log_probs).sum(dim=-1).mean()
    pred = gather.argmax(dim=-1)
    correct = (pred == resid_idx).sum()
    count = torch.tensor(resid_idx.numel(), device=logits.device, dtype=torch.long)
    return loss, correct, count


def compute_mark_losses(
    logits: Dict[str, Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor],
    helpers: Dict[str, FieldLossHelper],
    weights: LossWeights,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    weight_dict = weights.as_dict()
    total = None
    metrics: Dict[str, float] = {}

    def add_loss(key: str, loss: torch.Tensor, weight: float):
        nonlocal total
        if weight <= 0:
            metrics[f"{key}_loss"] = 0.0
            return
        if total is None:
            total = loss * weight
        else:
            total = total + loss * weight
        metrics[f"{key}_loss"] = float(loss.detach().item())

    def add_metric(key: str, value: torch.Tensor):
        metrics[key] = float(value.detach().item())

    for field in ("price", "size", "time", "type", "side", "level"):
        helper = helpers[field]
        field_logits = logits[field]
        coarse_targets = targets[f"{field}_coarse"]
        coarse_loss, coarse_correct, coarse_count = _semantic_ce(
            field_logits["coarse"], coarse_targets, helper.coarse_smoothing, ignore_index
        )
        add_loss(f"{field}_coarse", coarse_loss, weight_dict.get(f"{field}_coarse", 0.0))
        acc_value = (
            coarse_correct.float() / coarse_count.float() if coarse_count.item() > 0 else coarse_loss.new_zeros(())
        )
        add_metric(f"{field}_coarse_acc", acc_value)
        metrics[f"{field}_coarse_correct"] = float(coarse_correct.detach().item())
        metrics[f"{field}_coarse_count"] = float(coarse_count.detach().item())

        if field_logits["resid"] is not None:
            resid_targets = targets[f"{field}_resid"]
            resid_loss, resid_correct, resid_count = _semantic_ce_resid(
                field_logits["resid"], coarse_targets, resid_targets, helper, ignore_index
            )
            add_loss(f"{field}_resid", resid_loss, weight_dict.get(f"{field}_resid", 0.0))
            acc_value = (
                resid_correct.float() / resid_count.float() if resid_count.item() > 0 else resid_loss.new_zeros(())
            )
            add_metric(f"{field}_resid_acc", acc_value)
            metrics[f"{field}_resid_correct"] = float(resid_correct.detach().item())
            metrics[f"{field}_resid_count"] = float(resid_count.detach().item())
        else:
            metrics[f"{field}_resid_loss"] = 0.0
            metrics[f"{field}_resid_acc"] = 0.0
            metrics[f"{field}_resid_correct"] = 0.0
            metrics[f"{field}_resid_count"] = 0.0

    if total is None:
        total = next(iter(logits.values()))["coarse"].new_zeros(())
    metrics["mark_total_loss"] = float(total.detach().item())
    return total, metrics


def compute_tpp_loss(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    dt_next: torch.Tensor,
    censor: torch.Tensor,
    weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    log_intensity, Lambda = MixtureTPPHead.log_stats(alpha, beta, dt_next)
    event_mask = ~censor.bool()

    losses = []
    metrics: Dict[str, float] = {}

    if torch.any(event_mask):
        event_loss = (Lambda - log_intensity)[event_mask]
        losses.append(event_loss.mean())
        metrics["time_event_loss"] = float(event_loss.mean().detach().item())
    else:
        metrics["time_event_loss"] = 0.0

    if torch.any(censor):
        censor_loss = Lambda[censor]
        losses.append(censor_loss.mean())
        metrics["time_censor_loss"] = float(censor_loss.mean().detach().item())
    else:
        metrics["time_censor_loss"] = 0.0

    if not losses:
        total = alpha.new_zeros(())
    else:
        total = sum(losses)

    metrics["time_total_loss"] = float(total.detach().item())
    return total * weight, metrics


# =============================================================================
# Evaluation
# =============================================================================

def compute_calibration_bins(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int,
) -> List[Dict[str, float]]:
    probs = probs.view(-1)
    labels = labels.view(-1)
    bins: List[Dict[str, float]] = []
    if probs.numel() == 0:
        return bins
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=probs.device)
    for i in range(num_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == num_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = mask.sum()
        if count.item() == 0:
            continue
        pred_mean = probs[mask].mean().item()
        obs_rate = labels[mask].float().mean().item()
        bins.append({
            "bin_lower": float(lo.item()),
            "bin_upper": float(hi.item()),
            "pred_mean": pred_mean,
            "obs_rate": obs_rate,
            "count": float(count.item()),
        })
    return bins


def evaluate(
    model: HistoryModel,
    loader: DataLoader,
    device: torch.device,
    helpers: Dict[str, FieldLossHelper],
    loss_weights: LossWeights,
    tau_seconds: float,
    hier_maps: HierMaps,
    num_calib_bins: int = 10,
    ignore_index: int = -100,
    use_time_of_day: bool = False,
    use_soft_binning: bool = False,
) -> Dict[str, Any]:
    model.eval()
    total_time_loss = 0.0
    total_mark_loss = 0.0
    batches = 0
    correct_price = 0
    total_price = 0
    brier_sum = 0.0
    brier_count = 0
    prob_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []

    # Collect per-field metrics
    field_metrics: Dict[str, Dict[str, float]] = {
        field: {"coarse_correct": 0, "coarse_count": 0, "resid_correct": 0, "resid_count": 0}
        for field in ("price", "size", "time", "type", "side", "level")
    }

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
            dt_prev = batch["dt_prev"].to(device)
            dt_next = batch["dt_next"].to(device)
            censor = batch["censor"].to(device)
            time_abs = batch["time_absolute"].to(device) if use_time_of_day else None
            log_size_cont = batch["log_size_continuous"].to(device) if use_soft_binning else None
            dt_log_cont = batch["dt_log_continuous"].to(device) if use_soft_binning else None
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            (alpha, beta), mark_logits = model(inputs, dt_prev, time_abs, log_size_cont, dt_log_cont)
            time_loss, _ = compute_tpp_loss(alpha, beta, dt_next, censor, loss_weights.time_nll)
            mark_loss, mark_metrics = compute_mark_losses(mark_logits, targets, helpers, loss_weights, ignore_index)

            total_time_loss += float(time_loss.detach().item())
            total_mark_loss += float(mark_loss.detach().item())
            batches += 1

            # Accumulate per-field metrics
            for field in ("price", "size", "time", "type", "side", "level"):
                field_metrics[field]["coarse_correct"] += mark_metrics.get(f"{field}_coarse_correct", 0)
                field_metrics[field]["coarse_count"] += mark_metrics.get(f"{field}_coarse_count", 0)
                field_metrics[field]["resid_correct"] += mark_metrics.get(f"{field}_resid_correct", 0)
                field_metrics[field]["resid_count"] += mark_metrics.get(f"{field}_resid_count", 0)

            # Price accuracy on observed events
            price_mask = targets["price_coarse"] != ignore_index
            if price_mask.any():
                coarse_logits = mark_logits["price"]["coarse"]
                preds = coarse_logits.argmax(dim=-1)
                correct = (preds[price_mask] == targets["price_coarse"][price_mask]).sum().item()
                correct_price += correct
                total_price += int(price_mask.sum().item())

            # Brier score for P(event <= tau)
            probs = MixtureTPPHead.prob_within(alpha, beta, tau_seconds)
            labels = (~censor) & (dt_next <= tau_seconds)
            brier_sum += torch.mean((probs - labels.float()) ** 2).item()
            brier_count += 1
            prob_list.append(probs.detach().cpu().view(-1))
            label_list.append(labels.detach().cpu().float().view(-1))

    if batches == 0:
        return {"loss": float("inf")}

    result: Dict[str, Any] = {
        "time_loss": total_time_loss / batches,
        "mark_loss": total_mark_loss / batches,
        "loss": (total_time_loss + total_mark_loss) / batches,
    }

    # Add per-field accuracies
    for field, fm in field_metrics.items():
        if fm["coarse_count"] > 0:
            result[f"{field}_coarse_acc"] = fm["coarse_correct"] / fm["coarse_count"]
        if fm["resid_count"] > 0:
            result[f"{field}_resid_acc"] = fm["resid_correct"] / fm["resid_count"]

    if total_price > 0:
        result["price_coarse_acc"] = correct_price / total_price
    if brier_count > 0:
        result["brier_within_tau"] = brier_sum / brier_count
    if prob_list:
        probs_all = torch.cat(prob_list)
        labels_all = torch.cat(label_list)
        result["calibration_tau"] = compute_calibration_bins(probs_all, labels_all, max(1, int(num_calib_bins)))
    return result


# =============================================================================
# Optimizer and Scheduler
# =============================================================================

def make_optimizer(model: nn.Module, lr: float, betas: Tuple[float, float], weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)


class CosineWarmup:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, steps: int, peak_lr: float):
        self.opt = optimizer
        self.warmup = int(max(1, warmup))
        self.steps = int(max(self.warmup + 1, steps))
        self.peak = float(peak_lr)
        self.t = 0

    def step(self):
        self.t += 1
        if self.t <= self.warmup:
            lr = self.peak * (self.t / self.warmup)
        else:
            e = self.t - self.warmup
            T = max(1, self.steps - self.warmup)
            lr = 0.5 * (1 + math.cos(math.pi * e / T)) * self.peak
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


def build_loaders(
    feats: DerivedFeatures,
    token_bins: TokenBins,
    targets: NextEventTargets,
    dt_cap: float,
    tau_seconds: float,
    seq_cfg: SequenceConfig,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    N = len(token_bins.price_bins)
    train_end = int(N * 0.8)
    val_end = train_end + int(N * 0.1)
    slices = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
    }

    def slice_token_bins(sl: slice) -> TokenBins:
        return TokenBins(
            price_bins=token_bins.price_bins[sl],
            size_bins=token_bins.size_bins[sl],
            time_bins=token_bins.time_bins[sl],
            type_bins=token_bins.type_bins[sl],
            side_bins=token_bins.side_bins[sl],
            level_bins=token_bins.level_bins[sl],
        )

    def slice_targets(sl: slice) -> NextEventTargets:
        return NextEventTargets(
            price_coarse=targets.price_coarse[sl],
            price_resid=targets.price_resid[sl],
            size_coarse=targets.size_coarse[sl],
            size_resid=targets.size_resid[sl],
            time_coarse=targets.time_coarse[sl],
            time_resid=targets.time_resid[sl],
            type_coarse=targets.type_coarse[sl],
            type_resid=targets.type_resid[sl],
            side_coarse=targets.side_coarse[sl],
            side_resid=targets.side_resid[sl],
            level_coarse=targets.level_coarse[sl],
            level_resid=targets.level_resid[sl],
            has_event=targets.has_event[sl],
        )

    def slice_feats(sl: slice) -> DerivedFeatures:
        return DerivedFeatures(
            dp_ticks=feats.dp_ticks[sl],
            log_size=feats.log_size[sl],
            dt_log=feats.dt_log[sl],
            dt_prev=feats.dt_prev[sl],
            dt_next=feats.dt_next[sl],
            has_next_event=feats.has_next_event[sl],
            type_code=feats.type_code[sl],
            side_code=feats.side_code[sl],
            level_proxy=feats.level_proxy[sl],
            tick_size=feats.tick_size,
            time_scale=feats.time_scale,
            time_absolute=feats.time_absolute[sl],
        )

    loaders = {}
    for split, sl in slices.items():
        ds = HistoryDataset(
            slice_token_bins(sl),
            slice_targets(sl),
            slice_feats(sl),
            seq_cfg,
            dt_cap=dt_cap,
            tau_seconds=tau_seconds,
        )
        shuffle = split == "train"
        drop_last = shuffle and len(ds) >= batch_size
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_history,
        )
    return loaders["train"], loaders["val"]


# =============================================================================
# W&B Integration
# =============================================================================

def log_data_statistics(feats: DerivedFeatures, token_bins: TokenBins, spec: InputSpec):
    """Log dataset statistics to W&B."""
    # Create data statistics table
    data_table = wandb.Table(columns=["field", "num_bins", "min", "max", "mean", "std"])

    # Price bins
    data_table.add_data("price", spec.B_p,
                        int(token_bins.price_bins.min()),
                        int(token_bins.price_bins.max()),
                        float(token_bins.price_bins.mean()),
                        float(token_bins.price_bins.std()))

    # Size bins
    data_table.add_data("size", spec.B_s,
                        int(token_bins.size_bins.min()),
                        int(token_bins.size_bins.max()),
                        float(token_bins.size_bins.mean()),
                        float(token_bins.size_bins.std()))

    # Time bins
    data_table.add_data("time", spec.B_t,
                        int(token_bins.time_bins.min()),
                        int(token_bins.time_bins.max()),
                        float(token_bins.time_bins.mean()),
                        float(token_bins.time_bins.std()))

    # Type bins
    data_table.add_data("type", spec.B_ty,
                        int(token_bins.type_bins.min()),
                        int(token_bins.type_bins.max()),
                        float(token_bins.type_bins.mean()),
                        float(token_bins.type_bins.std()))

    # Side bins
    data_table.add_data("side", spec.B_si,
                        int(token_bins.side_bins.min()),
                        int(token_bins.side_bins.max()),
                        float(token_bins.side_bins.mean()),
                        float(token_bins.side_bins.std()))

    # Level bins
    data_table.add_data("level", spec.B_l,
                        int(token_bins.level_bins.min()),
                        int(token_bins.level_bins.max()),
                        float(token_bins.level_bins.mean()),
                        float(token_bins.level_bins.std()))

    wandb.log({"data/statistics": data_table}, step=0)

    # Log inter-arrival time statistics (step=0 for initial data stats)
    dt_next_valid = feats.dt_next[feats.has_next_event]
    wandb.log({
        "data/dt_next_mean": float(dt_next_valid.mean()),
        "data/dt_next_std": float(dt_next_valid.std()),
        "data/dt_next_median": float(np.median(dt_next_valid)),
        "data/total_events": len(feats.dt_next),
    }, step=0)


def log_calibration_curve(calibration_bins: List[Dict[str, float]], step: int):
    """Log calibration curve to W&B."""
    if not calibration_bins:
        return

    # Create calibration table
    calib_data = [[b["pred_mean"], b["obs_rate"], b["count"]] for b in calibration_bins]
    calib_table = wandb.Table(data=calib_data, columns=["predicted", "observed", "count"])

    wandb.log({
        "val/calibration_scatter": wandb.plot.scatter(
            calib_table, "predicted", "observed",
            title="Calibration: P(event within tau)"
        )
    }, step=step)


def save_model_artifact(model_path: str, config: Dict, best_metric: float, step: int):
    """Save model checkpoint as W&B artifact."""
    artifact = wandb.Artifact(
        name=f"mamba-history-{wandb.run.id}",
        type="model",
        description=f"Best checkpoint at step {step}",
        metadata={
            "val_mark_loss": best_metric,
            "step": step,
            "d_model": config.get("d_model"),
            "n_layers": config.get("n_layers"),
        }
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)


# =============================================================================
# Main Training Function
# =============================================================================

def train_with_wandb(config: Dict = None):
    """Main training function with W&B integration."""

    # Merge with defaults
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Initialize W&B
    run = wandb.init(
        config=cfg,
        save_code=True,
        tags=["mamba-history", "tpp"],
    )

    # Get config from wandb (allows sweep overrides)
    cfg = dict(wandb.config)

    # Set seed
    set_seed(cfg.get("seed", SEED_DEFAULT))

    # Select device
    device_str = cfg.get("device")
    if device_str is None:
        device = select_device()
    else:
        device = select_device(device_str)

    print(f"[wandb] Run: {wandb.run.name} ({wandb.run.id})")
    print(f"[device] Using {device}")

    # Load data
    print(f"[data] Loading from {cfg['data_path']}...")
    df = load_lobster_messages(cfg["data_path"])
    print(f"[data] Loaded {len(df)} events")

    feats_all = compute_derived_features(df, tick_size=cfg["tick_size"])

    # Fit bins on training data
    N = len(df)
    train_end = int(N * 0.8)
    train_slice = slice(0, train_end)

    be = BinEdges()
    feats_train = DerivedFeatures(
        dp_ticks=feats_all.dp_ticks[train_slice],
        log_size=feats_all.log_size[train_slice],
        dt_log=feats_all.dt_log[train_slice],
        dt_prev=feats_all.dt_prev[train_slice],
        dt_next=feats_all.dt_next[train_slice],
        has_next_event=feats_all.has_next_event[train_slice],
        type_code=feats_all.type_code[train_slice],
        side_code=feats_all.side_code[train_slice],
        level_proxy=feats_all.level_proxy[train_slice],
        tick_size=feats_all.tick_size,
        time_scale=feats_all.time_scale,
        time_absolute=feats_all.time_absolute[train_slice],
    )
    fb = fit_bins(feats_train, be)
    token_bins_all = compute_token_bins(feats_all, fb)

    spec = InputSpec(
        B_p=(be.dp_ticks_center_range[1] - be.dp_ticks_center_range[0] + 1) + 2,
        B_s=be.s_bins,
        B_t=be.t_bins,
        B_ty=be.type_bins,
        B_si=be.side_bins,
        B_l=be.level_bins,
    )

    # Log data statistics to W&B
    log_data_statistics(feats_all, token_bins_all, spec)

    # Build hierarchical maps
    hier_cfg = HierConfig()
    hier_maps = build_hier_maps(spec, hier_cfg, fb.dp_center_range)
    next_targets = make_next_event_targets(token_bins_all, hier_maps)

    # Build data loaders
    seq_cfg = SequenceConfig(
        seq_len=cfg.get("seq_len", 256),
        stride=cfg.get("stride", 128),
    )
    train_loader, val_loader = build_loaders(
        feats_all, token_bins_all, next_targets,
        dt_cap=cfg.get("dt_cap", 10.0),
        tau_seconds=cfg.get("tau", 0.75),
        seq_cfg=seq_cfg,
        batch_size=cfg.get("batch_size", 32),
    )

    if len(train_loader) == 0:
        raise RuntimeError("No training sequences produced. Reduce seq_len or increase data.")

    print(f"[data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    use_soft_binning = cfg.get("use_soft_binning", True)
    model = HistoryModel(
        spec=spec,
        hier_maps=hier_maps,
        d_model=cfg.get("d_model", 512),
        n_layers=cfg.get("n_layers", 6),
        dropout_emb=cfg.get("dropout_emb", 0.15),
        dropout_mlp=cfg.get("dropout_mlp", 0.32),
        mix_components=cfg.get("mix_components", 3),
        use_soft_binning=use_soft_binning,
        fitted_bins=fb if use_soft_binning else None,
    ).to(device)

    # Log model architecture
    wandb.watch(model, log="all", log_freq=100)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
    }, step=0)
    print(f"[model] Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Setup optimizer and scheduler
    opt = make_optimizer(
        model,
        lr=cfg.get("lr_peak", 0.0001),
        betas=(0.9, 0.95),
        weight_decay=cfg.get("weight_decay", 0.1),
    )
    sched = CosineWarmup(
        opt,
        warmup=cfg.get("warmup_steps", 400),
        steps=cfg.get("steps", 5000),
        peak_lr=cfg.get("lr_peak", 0.0001),
    )

    # Setup AMP
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Loss setup
    smoothing = SmoothingConfig()
    helpers = build_loss_helpers(hier_maps, smoothing, device)
    loss_weights = LossWeights()

    # Apply config overrides
    if cfg.get("time_nll"):
        loss_weights.time_nll = float(cfg["time_nll"])
    if cfg.get("mark_scale"):
        scale = float(cfg["mark_scale"])
        for k in list(loss_weights.as_dict().keys()):
            if k != "time_nll" and (k.endswith("coarse") or k.endswith("resid")):
                setattr(loss_weights, k, getattr(loss_weights, k) * scale)

    # Training loop
    use_time_of_day = cfg.get("use_time_of_day", True)
    tau_seconds = cfg.get("tau", 0.75)
    steps = cfg.get("steps", 5000)
    patience = cfg.get("patience", 10)
    grad_clip = cfg.get("grad_clip", 1.0)

    start_time = time.time()
    step = 0
    best_val = None
    best_metric = float("inf")
    no_improve = 0
    stop_training = False

    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)

    model.train()

    print(f"[train] Starting training for {steps} steps...")

    while step < steps and not stop_training:
        epoch_batches = 0
        for batch in train_loader:
            epoch_batches += 1
            step += 1

            inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
            dt_prev = batch["dt_prev"].to(device)
            dt_next = batch["dt_next"].to(device)
            censor = batch["censor"].to(device)
            time_abs = batch["time_absolute"].to(device) if use_time_of_day else None
            log_size_cont = batch["log_size_continuous"].to(device) if use_soft_binning else None
            dt_log_cont = batch["dt_log_continuous"].to(device) if use_soft_binning else None
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            try:
                autocast_ctx = torch.amp.autocast(amp_device, enabled=amp_enabled)
            except AttributeError:
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled)

            with autocast_ctx:
                (alpha, beta), mark_logits = model(inputs, dt_prev, time_abs, log_size_cont, dt_log_cont)
                time_loss, time_metrics = compute_tpp_loss(alpha, beta, dt_next, censor, loss_weights.time_nll)
                mark_loss, mark_metrics = compute_mark_losses(mark_logits, targets, helpers, loss_weights)
                loss = time_loss + mark_loss

            scaler.scale(loss).backward()

            # Gradient clipping
            if grad_clip > 0:
                scaler.unscale_(opt)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            else:
                grad_norm = 0.0

            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            lr = sched.step()

            # Log training metrics
            if step % 20 == 0 or step == 1:
                log_dict = {
                    "train/step": step,
                    "train/loss": loss.item(),
                    "train/time_loss": time_metrics["time_total_loss"],
                    "train/mark_loss": mark_metrics["mark_total_loss"],
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                }

                # Per-field accuracies
                for field in ("price", "size", "time", "type", "side", "level"):
                    log_dict[f"train/{field}_coarse_acc"] = mark_metrics.get(f"{field}_coarse_acc", 0.0)

                wandb.log(log_dict, step=step)

                print(f"step {step} | loss={loss.item():.4f} | time={time_metrics['time_total_loss']:.4f} | mark={mark_metrics['mark_total_loss']:.4f} | lr={lr:.2e}")

            if step >= steps:
                break

        # Validation at end of each epoch
        if epoch_batches > 0:
            val_metrics = evaluate(
                model, val_loader, device, helpers, loss_weights,
                tau_seconds, hier_maps,
                num_calib_bins=cfg.get("calibration_bins", 10),
                use_time_of_day=use_time_of_day,
                use_soft_binning=use_soft_binning,
            )

            score = val_metrics.get("mark_loss", float("inf"))

            # Log validation metrics
            val_log = {
                "train/step": step,
                "val/loss": val_metrics.get("loss", float("inf")),
                "val/time_loss": val_metrics.get("time_loss", 0.0),
                "val/mark_loss": val_metrics.get("mark_loss", 0.0),
                "val/brier_within_tau": val_metrics.get("brier_within_tau", 0.0),
            }

            # Per-field validation accuracies
            for field in ("price", "size", "time", "type", "side", "level"):
                if f"{field}_coarse_acc" in val_metrics:
                    val_log[f"val/{field}_coarse_acc"] = val_metrics[f"{field}_coarse_acc"]

            wandb.log(val_log, step=step)

            # Log calibration curve
            if "calibration_tau" in val_metrics:
                log_calibration_curve(val_metrics["calibration_tau"], step)

            print(f"[val] step={step} | loss={val_metrics.get('loss', float('inf')):.4f} | mark={score:.4f} | brier={val_metrics.get('brier_within_tau', 0.0):.4f}")

            # Check for improvement
            if score < best_metric:
                best_metric = score
                best_val = val_metrics
                no_improve = 0

                # Save checkpoint
                checkpoint_path = "artifacts/mamba_history_wandb_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "step": step,
                    "best_metric": best_metric,
                }, checkpoint_path)

                # Save as W&B artifact
                save_model_artifact(checkpoint_path, cfg, best_metric, step)

                print(f"[checkpoint] New best model saved (mark_loss={score:.4f})")

                # Alert for new best
                try:
                    wandb.alert(
                        title="New Best Model",
                        text=f"Val mark_loss improved to {score:.4f} at step {step}",
                        level=wandb.AlertLevel.INFO,
                    )
                except Exception:
                    pass  # Alerts may not be available in all configurations
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    print(f"[early-stop] No improvement for {patience} evaluations. Stopping.")
                    stop_training = True

        if stop_training:
            break

    # Training complete
    total_minutes = (time.time() - start_time) / 60.0

    # Log final summary
    wandb.run.summary["best_val_mark_loss"] = best_metric
    wandb.run.summary["best_val_loss"] = best_val.get("loss") if best_val else float("inf")
    wandb.run.summary["total_steps"] = step
    wandb.run.summary["total_minutes"] = total_minutes
    wandb.run.summary["early_stopped"] = stop_training

    print(f"[done] Training complete in {total_minutes:.1f} min")
    print(f"[done] Best mark_loss: {best_metric:.4f}")

    wandb.finish()

    return {
        "best_metric": best_metric,
        "best_val": best_val,
        "steps_ran": step,
        "elapsed_minutes": total_minutes,
    }


# =============================================================================
# Sweep Functions
# =============================================================================

def create_sweep(project: str, quick: bool = False) -> str:
    """Create a W&B sweep and return the sweep ID."""
    sweep_config = SWEEP_CONFIG_QUICK if quick else SWEEP_CONFIG
    sweep_id = wandb.sweep(sweep_config, project=project)
    print(f"[sweep] Created sweep: {sweep_id}")
    print(f"[sweep] View at: https://wandb.ai/{project}/sweeps/{sweep_id}")
    return sweep_id


def run_sweep_agent(sweep_id: str, project: str, count: int = 30):
    """Run a W&B sweep agent."""
    print(f"[sweep] Running {count} trials for sweep {sweep_id}")

    def sweep_train():
        train_with_wandb()

    wandb.agent(sweep_id, function=sweep_train, project=project, count=count)


# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mamba History Model Training with W&B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single training run with defaults
  python train_wandb.py train --project mamba-history

  # Training with custom hyperparameters
  python train_wandb.py train --project mamba-history --d-model 384 --n-layers 5

  # Create a hyperparameter sweep
  python train_wandb.py sweep create --project mamba-history

  # Run sweep agent (30 trials)
  python train_wandb.py sweep run --sweep-id <sweep-id> --project mamba-history --count 30
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run a single training")
    train_parser.add_argument("--project", type=str, default="mamba-history", help="W&B project name")
    train_parser.add_argument("--name", type=str, default=None, help="W&B run name")
    train_parser.add_argument("--data-path", type=str, default=None, help="Path to LOBSTER message CSV")
    train_parser.add_argument("--d-model", type=int, default=None, help="Model dimension")
    train_parser.add_argument("--n-layers", type=int, default=None, help="Number of Mamba layers")
    train_parser.add_argument("--dropout-emb", type=float, default=None, help="Embedding dropout")
    train_parser.add_argument("--dropout-mlp", type=float, default=None, help="MLP dropout")
    train_parser.add_argument("--lr-peak", type=float, default=None, help="Peak learning rate")
    train_parser.add_argument("--steps", type=int, default=None, help="Training steps")
    train_parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    train_parser.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    train_parser.add_argument("--time-nll", type=float, default=None, help="Time NLL weight")
    train_parser.add_argument("--mark-scale", type=float, default=None, help="Mark loss scale")
    train_parser.add_argument("--mix-components", type=int, default=None, help="Mixture components")
    train_parser.add_argument("--tau", type=float, default=None, help="Prediction horizon (seconds)")
    train_parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    train_parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None)
    train_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    train_parser.add_argument("--no-time-of-day", action="store_true", help="Disable time-of-day encoding")
    train_parser.add_argument("--no-soft-binning", action="store_true", help="Disable soft binning")

    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Hyperparameter sweep")
    sweep_subparsers = sweep_parser.add_subparsers(dest="sweep_command", help="Sweep subcommand")

    # Sweep create
    create_parser = sweep_subparsers.add_parser("create", help="Create a new sweep")
    create_parser.add_argument("--project", type=str, required=True, help="W&B project name")
    create_parser.add_argument("--quick", action="store_true", help="Use quick grid search config")

    # Sweep run
    run_parser = sweep_subparsers.add_parser("run", help="Run sweep agent")
    run_parser.add_argument("--sweep-id", type=str, required=True, help="Sweep ID")
    run_parser.add_argument("--project", type=str, required=True, help="W&B project name")
    run_parser.add_argument("--count", type=int, default=30, help="Number of trials to run")

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.command == "train":
        # Build config from CLI args
        config = {}

        if args.data_path:
            config["data_path"] = args.data_path
        if args.d_model:
            config["d_model"] = args.d_model
        if args.n_layers:
            config["n_layers"] = args.n_layers
        if args.dropout_emb:
            config["dropout_emb"] = args.dropout_emb
        if args.dropout_mlp:
            config["dropout_mlp"] = args.dropout_mlp
        if args.lr_peak:
            config["lr_peak"] = args.lr_peak
        if args.steps:
            config["steps"] = args.steps
        if args.warmup_steps:
            config["warmup_steps"] = args.warmup_steps
        if args.batch_size:
            config["batch_size"] = args.batch_size
        if args.seq_len:
            config["seq_len"] = args.seq_len
        if args.time_nll:
            config["time_nll"] = args.time_nll
        if args.mark_scale:
            config["mark_scale"] = args.mark_scale
        if args.mix_components:
            config["mix_components"] = args.mix_components
        if args.tau:
            config["tau"] = args.tau
        if args.patience:
            config["patience"] = args.patience
        if args.device:
            config["device"] = args.device
        if args.seed:
            config["seed"] = args.seed
        if args.no_time_of_day:
            config["use_time_of_day"] = False
        if args.no_soft_binning:
            config["use_soft_binning"] = False

        # Set W&B project and run name
        os.environ["WANDB_PROJECT"] = args.project
        if args.name:
            os.environ["WANDB_NAME"] = args.name

        # Run training
        train_with_wandb(config)

    elif args.command == "sweep":
        if args.sweep_command == "create":
            create_sweep(args.project, quick=args.quick)
        elif args.sweep_command == "run":
            run_sweep_agent(args.sweep_id, args.project, count=args.count)
        else:
            parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
