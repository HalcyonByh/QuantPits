import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    (workspace / "output").mkdir()
    
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, run_rolling_analysis
    import importlib
    importlib.reload(env)
    
    # Reload run_rolling_analysis to pick up reloaded env
    importlib.reload(run_rolling_analysis)
    
    # Patch env.ROOT_DIR in the context of run_rolling_analysis
    monkeypatch.setattr(run_rolling_analysis.env, 'ROOT_DIR', str(workspace))
    monkeypatch.setattr(env, 'ROOT_DIR', str(workspace))
    
    yield run_rolling_analysis, workspace

@patch('quantpits.scripts.run_rolling_analysis.init_qlib')
@patch('quantpits.scripts.run_rolling_analysis.load_market_config')
@patch('quantpits.scripts.run_rolling_analysis.get_daily_features')
@patch('quantpits.scripts.run_rolling_analysis.PortfolioAnalyzer')
@patch('quantpits.scripts.run_rolling_analysis.ExecutionAnalyzer')
def test_compute_rolling_metrics(mock_exec, mock_port, mock_features, mock_market, mock_init, mock_env):
    rra, workspace = mock_env
    
    mock_market.return_value = ("csi300", "SH000300")
    
    # Mock portfolio returns
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    returns_df = pd.Series(np.random.normal(0.001, 0.02, 40), index=dates, name="Portfolio")
    
    mock_pa = MagicMock()
    mock_port.return_value = mock_pa
    mock_pa.calculate_daily_returns.return_value = returns_df
    mock_pa.daily_amount = pd.DataFrame({"CSI300": np.linspace(1000, 1100, 40)}, index=dates)
    
    # Mock market features with multiple instruments
    instruments = ["A", "B", "C", "D", "E"]
    feat_idx = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "datetime"])
    n = len(feat_idx)
    mock_feat_df = pd.DataFrame({
        "close": np.random.uniform(10, 20, n),
        "volume": np.random.uniform(1000, 2000, n)
    }, index=feat_idx)
    mock_features.return_value = mock_feat_df
    
    # Mock execution analyzer
    mock_ex = MagicMock()
    mock_exec.return_value = mock_ex
    mock_ex.calculate_slippage_and_delay.return_value = pd.DataFrame({
        "Exec_Slippage": [0.001] * 20, 
        "Delay_Cost": [0.0005] * 20, 
        "Total_Friction": [0.0015] * 20,
        "成交日期": dates[:20]
    })
    mock_ex.trade_log = pd.DataFrame({
        "证券代码": ["A"] * 2,
        "成交日期": [dates[0], dates[2]],
        "交易类别": ["买入", "卖出"],
        "成交价格": [10.0, 11.0],
        "成交数量": [100, 100]
    })
    
    # Run with windows
    rra.compute_rolling_metrics(windows=[5], sub_window=5, market='csi300')
    
    out_file = workspace / "output" / "rolling_metrics_5.csv"
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert "Portfolio_Return" in df.columns
    assert "Exposure_Market_Beta" in df.columns

def test_compute_rolling_metrics_no_returns(mock_env):
    rra, workspace = mock_env
    # We need to mock PortfolioAnalyzer here too, but since the mock_env Reloaded run_rolling_analysis,
    # let's just patch it in the test.
    with patch('quantpits.scripts.run_rolling_analysis.PortfolioAnalyzer') as mock_port:
        mock_pa = MagicMock()
        mock_port.return_value = mock_pa
        mock_pa.calculate_daily_returns.return_value = pd.Series(dtype=float)
        
        rra.compute_rolling_metrics(windows=[2])
        
        out_file = workspace / "output" / "rolling_metrics_2.csv"
        assert not out_file.exists()
