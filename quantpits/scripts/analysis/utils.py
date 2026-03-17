import os
import json
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)
from quantpits.utils import env

ROOT_DIR = env.ROOT_DIR
CONFIG_FILE = os.path.join(ROOT_DIR, "config", "prod_config.json")
MODEL_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "model_config.json")
PREDICTION_DIR = os.path.join(ROOT_DIR, "output", "predictions")

# 默认市场和基准（当配置文件不存在时使用）
DEFAULT_MARKET = "csi300"
DEFAULT_BENCHMARK = "SH000300"


def load_market_config():
    """
    使用 config_loader 加载统一配置并返回 market 和 benchmark。
    """
    from quantpits.utils.config_loader import load_workspace_config
    try:
        config = load_workspace_config(ROOT_DIR)
        
        market = config.get("market", DEFAULT_MARKET)
        benchmark = config.get("benchmark", DEFAULT_BENCHMARK)
        return market, benchmark
    except Exception as e:
        print(f"Warning: Failed to load market config: {e}")
        return DEFAULT_MARKET, DEFAULT_BENCHMARK


def init_qlib():
    """Initialize Qlib environment (delegates to env.init_qlib)."""
    env.init_qlib()


def get_trading_dates(start_date, end_date):
    """Get market trading dates between start_date and end_date."""
    from qlib.data import D
    cal = D.calendar(start_time=start_date, end_time=end_date)
    return [d.strftime('%Y-%m-%d') for d in cal]


def get_daily_features(start_date, end_date, market=None, features=None):
    """
    Fetch daily features from Qlib. Default fetches price information.
    market 默认从 model_config.json 读取，未配置时 fallback 到 csi300。
    """
    from qlib.data import D
    if market is None:
        market, _ = load_market_config()
    if features is None:
        features = {
            'close': '$close',
            'open': '$open',
            'high': '$high',
            'low': '$low',
            'vwap': '$vwap',
            'factor': '$factor',
            'adj_close': '$close / $factor',
            'adj_open': '$open / $factor',
        }
    instruments = D.instruments(market=market)
    fields = list(features.values())
    names = list(features.keys())
    
    df = D.features(instruments, fields, start_time=start_date, end_time=end_date)
    df.columns = names
    return df


def get_forward_returns(start_date, end_date, market=None, n_days=1):
    """
    Fetch N-day forward returns from Qlib.
    Return from T's close to T+N's close.
    market 默认从 model_config.json 读取。
    """
    from qlib.data import D
    if market is None:
        market, _ = load_market_config()
    instruments = D.instruments(market=market)
    field = f"Ref($close, -{n_days}) / $close - 1"
    df = D.features(instruments, [field], start_time=start_date, end_time=end_date)
    df.columns = [f'return_{n_days}d']
    return df


def load_trade_log():
    """Load the realistic full trade execution log."""
    trade_log_path = os.path.join(ROOT_DIR, "data", "trade_log_full.csv")
    if not os.path.exists(trade_log_path):
        print(f"Warning: Trade log not found at {trade_log_path}")
        return pd.DataFrame()
    df = pd.read_csv(trade_log_path)
    if '成交日期' in df.columns:
        df['成交日期'] = pd.to_datetime(df['成交日期'])
    return df


def load_daily_amount():
    """Load the daily NAV and portfolio amount data."""
    daily_amount_path = os.path.join(ROOT_DIR, "data", "daily_amount_log_full.csv")
    if not os.path.exists(daily_amount_path):
        print(f"Warning: Daily amount log not found at {daily_amount_path}")
        return pd.DataFrame()
    df = pd.read_csv(daily_amount_path)
    if '成交日期' in df.columns:
        df['成交日期'] = pd.to_datetime(df['成交日期'])
    return df

def load_holding_log():
    """Load the daily holding log."""
    holding_log_path = os.path.join(ROOT_DIR, "data", "holding_log_full.csv")
    if not os.path.exists(holding_log_path):
        print(f"Warning: Holding log not found at {holding_log_path}")
        return pd.DataFrame()
    df = pd.read_csv(holding_log_path)
    if '成交日期' in df.columns:
        df['成交日期'] = pd.to_datetime(df['成交日期'])
    return df


def load_model_predictions(model_name, start_date=None, end_date=None):
    """
    Load a model's prediction series from Qlib Recorders spanning the requested dates.
    Here we expect the model record id to be found in config/latest_train_records.json or config/latest_rolling_records.json
    """
    from qlib.workflow import R
    
    # Try looking in latest_train_records.json
    record_id, experiment_name = None, None
    for rec_file in ['latest_train_records.json', 'latest_rolling_records.json']:
        file_path = os.path.join(ROOT_DIR, "config", rec_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    rec_data = json.load(f)
                if model_name in rec_data.get('models', {}):
                    record_id = rec_data['models'][model_name]
                    experiment_name = rec_data.get('experiment_name')
                    break
            except Exception:
                pass
                
    if not record_id:
        print(f"Warning: Model {model_name} not found in train/rolling records.")
        return pd.DataFrame()
        
    try:
        recorder = R.get_recorder(recorder_id=record_id, experiment_name=experiment_name)
        df = recorder.load_object("pred.pkl")
    except Exception as e:
        print(f"Error loading pred.pkl from recorder {record_id} for {model_name}: {e}")
        return pd.DataFrame()
        
    if isinstance(df, pd.Series):
        df = df.to_frame('score')
        
    if '0' in df.columns and 'score' not in df.columns:
        df = df.rename(columns={'0': 'score'})
    # Normalize the column names
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if 'score' not in df.columns and num_cols:
        df = df.rename(columns={num_cols[0]: 'score'})
        
    concat_df = df
    if 'datetime' in concat_df.columns:
        concat_df['datetime'] = pd.to_datetime(concat_df['datetime'])
        concat_df = concat_df.set_index(['datetime', 'instrument']).sort_index()
    else:
        # If it doesn't have datetime as column, it might be in index
        if concat_df.index.nlevels == 2 and 'datetime' in concat_df.index.names:
            concat_df = concat_df.sort_index()
            
    # Drop duplicates if multiple prediction files have overlapping days
    concat_df = concat_df[~concat_df.index.duplicated(keep='last')]
    
    if start_date:
        concat_df = concat_df[concat_df.index.get_level_values('datetime') >= pd.to_datetime(start_date)]
    if end_date:
        concat_df = concat_df[concat_df.index.get_level_values('datetime') <= pd.to_datetime(end_date)]
            
    return concat_df

