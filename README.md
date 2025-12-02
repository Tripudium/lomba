# LOB Mamba - Limit Order Book Modeling with Mamba

A high-performance temporal point process model for limit order book data using Mamba architecture.

## Overview

This project implements a state-space model for predicting next-event distributions in limit order books, combining:
- **Mamba architecture** for efficient sequence modeling
- **Temporal Point Process (TPP)** heads for time prediction
- **Hierarchical mark prediction** for price, size, type, side, and level

## Installation

```bash
# Install dependencies using uv
uv sync

# Or with pip
pip install -r requirements.txt
```

## Data Loading

### Quick Start

The unified dataloader supports multiple exchanges (MIC, LOBSTER) with a simple timestamp-based API:

```python
from main import load_messages, validate_messages

# Load data using timestamps (YYMMDD.HHMMSS format)
df = load_messages(
    product="AAPL",
    source="lobster",
    times=["200622.093000", "200622.160000"]  # June 22, 2020, 9:30 AM to 4:00 PM
)

# Validate data quality
validate_messages(df, source="lobster")
```

### Timestamp Format

Timestamps use `YYMMDD.HHMMSS` format:
- `200622.093000` = June 22, 2020 at 09:30:00
- `200622.160000` = June 22, 2020 at 16:00:00

### Supported Data Sources

#### LOBSTER
- **Path**: `data/lobster/`
- **Format**: `{product}_{date}_messages.parquet`
- **Columns**: `tst`, `event_code`, `vol`, `prc`, `direction`, `order_id`
- **Time unit**: nanoseconds

#### MIC
- **Path**: `data/mic/`
- **Format**: `{date}__RTSX_FUT__{product}.parquet`
- **Columns**: `tst`, `event_code`, `vol`, `prc`, `is_buy`, `order_id`
- **Time unit**: nanoseconds

### Advanced Usage

#### Custom Exchange Configuration

```python
from main import load_messages, ExchangeConfig

custom_config = ExchangeConfig(
    data_path="data/custom",
    filename_template="{product}_{date}.parquet",
    date_format="%Y-%m-%d",
    column_mapping={
        "timestamp": "time",
        "event_type": "type",
        "volume": "size",
        "price": "price",
        "side": "direction",
    },
    time_unit="ms",  # milliseconds
    price_scale=1.0
)

df = load_messages(
    product="CUSTOM", 
    source="custom",
    times=["200601.000000", "200603.235959"],
    config=custom_config
)
```

## Training

### Basic Training

```bash
# Train with default configuration
python main.py train --project mamba-lob

# Train with custom hyperparameters
python main.py train --project mamba-lob \
    --d-model 384 \
    --n-layers 5 \
    --batch-size 32 \
    --lr-peak 0.0001
```

### Configuration

Update `DEFAULT_CONFIG` in `main.py` or pass via command line:

```python
DEFAULT_CONFIG = {
    # Model architecture
    "d_model": 512,          # Model dimension
    "n_layers": 6,           # Number of Mamba layers
    "dropout_emb": 0.15,     # Embedding dropout
    "dropout_mlp": 0.32,     # MLP dropout
    
    # Training
    "lr_peak": 0.0001,       # Peak learning rate
    "steps": 5000,           # Training steps
    "warmup_steps": 400,     # Warmup steps
    "batch_size": 32,        # Batch size
    "seq_len": 256,          # Sequence length
    "stride": 128,           # Stride for sequences
    
    # Data
    "data_config": {
        "source": "lobster",
        "product": "AAPL",
        "start_time": "200601.000000",
        "end_time": "200603.235959",
    },
    "tick_size": 0.01,       # Price tick size
}
```

### Hyperparameter Sweeps

```bash
# Create a sweep
python main.py sweep create --project mamba-lob

# Run sweep agent
python main.py sweep run --sweep-id <sweep-id> --project mamba-lob --count 30
```

## Model Architecture

### Input Features
- **Price**: Discrete price changes in ticks
- **Size**: Log-transformed order size
- **Time**: Log inter-arrival time
- **Type**: Event type (add/cancel/execute/hidden)
- **Side**: Buy/sell indicator
- **Level**: Price level proxy

### Output Predictions
- **Time**: Next event time via mixture of exponentials
- **Marks**: Hierarchical prediction of price, size, type, side, level

### Training Objective
```
Loss = λ_time * TPP_loss + Σ(λ_mark * CrossEntropy_mark)
```

## Data Pipeline

1. **Load**: Read parquet files for date range
2. **Validate**: Check for nulls, sorting, invalid values
3. **Feature Engineering**: Compute price changes, inter-arrival times
4. **Binning**: Quantile-based discretization
5. **Hierarchical Mapping**: Build coarse/fine token structures
6. **Sequence Generation**: Create overlapping sequences

## Performance

- **Polars** for fast DataFrame operations
- **Parquet** for efficient storage
- **Automatic mixed precision** (AMP) on CUDA
- **Gradient accumulation** for large batches

## Monitoring

All experiments tracked with Weights & Biases:
- Training/validation metrics
- Calibration curves
- Model artifacts
- Hyperparameter configs

## File Structure

```
.
├── main.py                 # Main training script with dataloader
├── data/
│   ├── lobster/           # LOBSTER data files
│   └── mic/               # MIC data files
├── artifacts/             # Model checkpoints
└── examples/              # Example scripts
```

## Example Workflow

```python
import polars as pl
from main import (
    load_messages, 
    validate_messages,
    compute_derived_features,
    fit_bins,
    compute_token_bins
)

# 1. Load data
df = load_messages(
    product="AAPL",
    source="lobster", 
    times=["200622.093000", "200622.160000"]
)

# 2. Validate
validate_messages(df, source="lobster")

# 3. Compute features
features = compute_derived_features(df, tick_size=0.01)

# 4. Fit bins on training data
bins = fit_bins(features, BinEdges())

# 5. Discretize
tokens = compute_token_bins(features, bins)

# 6. Train model (see main.py train_with_wandb)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lob_mamba,
  title={LOB Mamba: Temporal Point Process Modeling for Limit Order Books},
  year={2024},
  url={https://github.com/...}
}
```

## License

MIT License - see LICENSE file for details
