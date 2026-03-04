import pytest
import pandas as pd
import numpy as np
import os
import json
from unittest.mock import patch, MagicMock

from quantpits.scripts.analysis import utils as ut


def test_load_market_config(tmp_path):
    # Test fallback
    with patch('quantpits.scripts.analysis.utils.MODEL_CONFIG_FILE', str(tmp_path / "missing.json")):
        market, benchmark = ut.load_market_config()
        assert market == ut.DEFAULT_MARKET
        assert benchmark == ut.DEFAULT_BENCHMARK
    
    # Test parsed config
    config_path = tmp_path / "model_config.json"
    config_path.write_text(json.dumps({
        "market": "my_market",
        "benchmark": "my_benchmark"
    }))
    
    with patch('quantpits.scripts.analysis.utils.MODEL_CONFIG_FILE', str(config_path)):
        market, benchmark = ut.load_market_config()
        assert market == "my_market"
        assert benchmark == "my_benchmark"
        
    # Test corrupted parsed config
    config_path.write_text("not a valid json")
    with patch('quantpits.scripts.analysis.utils.MODEL_CONFIG_FILE', str(config_path)):
        market, benchmark = ut.load_market_config()
        assert market == ut.DEFAULT_MARKET
        assert benchmark == ut.DEFAULT_BENCHMARK


@patch('quantpits.scripts.analysis.utils.env.init_qlib')
def test_init_qlib(mock_init):
    ut.init_qlib()
    mock_init.assert_called_once()


def test_get_trading_dates():
    with patch('qlib.data.D') as mock_d:
        # Mock the calendar return to simulate pd.date_range
        dates = pd.date_range("2026-01-01", "2026-01-03")
        mock_d.calendar.return_value = dates
        
        result = ut.get_trading_dates("2026-01-01", "2026-01-03")
        assert result == ["2026-01-01", "2026-01-02", "2026-01-03"]
        mock_d.calendar.assert_called_once_with(start_time="2026-01-01", end_time="2026-01-03")


def test_get_daily_features():
    with patch('qlib.data.D') as mock_d:
        mock_d.instruments.return_value = "instruments"
        
        df = pd.DataFrame({
            "col1": [1.0, 2.0],
            "col2": [3.0, 4.0]
        })
        mock_d.features.return_value = df
        
        features_map = {"my_close": "$close", "my_open": "$open"}
        result = ut.get_daily_features("2026-01-01", "2026-01-02", market="test_market", features=features_map)
        
        mock_d.instruments.assert_called_once_with(market="test_market")
        mock_d.features.assert_called_once_with("instruments", ["$close", "$open"], start_time="2026-01-01", end_time="2026-01-02")
        
        assert list(result.columns) == ["my_close", "my_open"]


def test_get_forward_returns():
    with patch('qlib.data.D') as mock_d:
        mock_d.instruments.return_value = "instruments"
        
        df = pd.DataFrame({
            "ret": [0.05, -0.01]
        })
        mock_d.features.return_value = df
        
        result = ut.get_forward_returns("2026-01-01", "2026-01-02", market="test_market", n_days=5)
        
        mock_d.instruments.assert_called_once_with(market="test_market")
        mock_d.features.assert_called_once_with("instruments", ["Ref($close, -5) / $close - 1"], start_time="2026-01-01", end_time="2026-01-02")
        
        assert list(result.columns) == ["return_5d"]


# ── CSV Loaders ──────────────────────────────────────────────────────────

def test_load_csvs(tmp_path):
    # Setup dummy files
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    trade = "成交日期,证券代码\n2026-01-01,SZ000001\n"
    daily = "成交日期,收盘价值\n2026-01-01,1000\n"
    holding = "成交日期,证券代码\n2026-01-01,CASH\n"
    
    (data_dir / "trade_log_full.csv").write_text(trade)
    (data_dir / "daily_amount_log_full.csv").write_text(daily)
    (data_dir / "holding_log_full.csv").write_text(holding)
    
    with patch('quantpits.scripts.analysis.utils.ROOT_DIR', str(tmp_path)):
        # Test existing
        t_df = ut.load_trade_log()
        assert len(t_df) == 1
        assert t_df.iloc[0]["证券代码"] == "SZ000001"
        assert pd.api.types.is_datetime64_any_dtype(t_df["成交日期"])
        
        d_df = ut.load_daily_amount()
        assert len(d_df) == 1
        assert d_df.iloc[0]["收盘价值"] == 1000
        assert pd.api.types.is_datetime64_any_dtype(d_df["成交日期"])
        
        h_df = ut.load_holding_log()
        assert len(h_df) == 1
        assert h_df.iloc[0]["证券代码"] == "CASH"
        
        # Test missing by pointing to an empty dir
        with patch('quantpits.scripts.analysis.utils.ROOT_DIR', str(tmp_path / "empty_dir")):
            assert ut.load_trade_log().empty
            assert ut.load_daily_amount().empty
            assert ut.load_holding_log().empty


# ── load_model_predictions ───────────────────────────────────────────────

def test_load_model_predictions(tmp_path):
    pred_dir = tmp_path / "output" / "predictions"
    pred_dir.mkdir(parents=True)
    
    # 1. First file with '0' as score column, missing datetime in cols (index instead)
    # We write a normal CSV, then we'll pretend index parsing behavior when loaded?
    # Actually utils.py loads normal CSV, so datetime will become a column.
    df1 = pd.DataFrame({
        "datetime": ["2026-01-01", "2026-01-02"],
        "instrument": ["SZ000001", "SZ000001"],
        "0": [0.5, 0.6]
    })
    df1.to_csv(pred_dir / "modelA_2026-01-02.csv", index=False)
    
    # 2. Second file with 'score' as column
    df2 = pd.DataFrame({
        "datetime": ["2026-01-02", "2026-01-03"],
        "instrument": ["SZ000001", "SZ000001"],
        "score": [0.65, 0.7]
    })
    df2.to_csv(pred_dir / "modelA_2026-01-03.csv", index=False)
    
    # 3. Third file for a different model entirely
    df3 = pd.DataFrame({
        "datetime": ["2026-01-03"],
        "instrument": ["SZ000001"],
        "score": [0.9]
    })
    df3.to_csv(pred_dir / "modelB_2026-01-03.csv", index=False)
    
    with patch('quantpits.scripts.analysis.utils.PREDICTION_DIR', str(pred_dir)):
        # Test basic loading and duplicate dropping (2026-01-02 is in both, last one kept)
        df_a = ut.load_model_predictions("modelA")
        assert not df_a.empty
        assert len(df_a) == 3 # 01, 02, 03
        
        # Ensure '0' was renamed to 'score'
        assert 'score' in df_a.columns
        
        # Ensure 01-02 took the latest value (0.65)
        # Because the index is a MultiIndex [datetime, instrument].
        # We need to test the value for ('2026-01-02', 'SZ000001')
        val = df_a.loc[(pd.Timestamp("2026-01-02"), "SZ000001"), "score"]
        if isinstance(val, pd.Series):
            val = val.iloc[-1]
        assert np.isclose(float(val), 0.65)
        
        # Test date filtering
        df_a_filtered = ut.load_model_predictions("modelA", start_date="2026-01-02", end_date="2026-01-02")
        assert len(df_a_filtered) == 1
        
        # Test missing model
        assert ut.load_model_predictions("modelC").empty
