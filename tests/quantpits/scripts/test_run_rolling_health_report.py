import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    out_dir = workspace / "output"
    out_dir.mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, run_rolling_health_report
    import importlib
    importlib.reload(env)
    monkeypatch.setattr(run_rolling_health_report.env, 'ROOT_DIR', str(workspace))
    
    # Needs to re-read what ROOT_DIR is because evaluate_health constructs the path dynamically
    # Note: run_rolling_health_report imports env without from, so env.ROOT_DIR works
    yield run_rolling_health_report, out_dir

def test_evaluate_health_missing_files(mock_env, capsys):
    rhr, out_dir = mock_env
    # No CSVs created
    rhr.evaluate_health()
    
    captured = capsys.readouterr()
    assert "Required rolling metrics CSVs not found!" in captured.out

def test_evaluate_health_insufficient_data(mock_env, capsys):
    rhr, out_dir = mock_env
    
    # Create 20-day dummy (small data)
    dates = pd.date_range("2020-01-01", "2020-01-10")
    df = pd.DataFrame({"Date": dates, "Exec_Slippage_Mean": [0]*10})
    df.to_csv(out_dir / "rolling_metrics_20.csv", index=False)
    df.to_csv(out_dir / "rolling_metrics_60.csv", index=False)
    
    rhr.evaluate_health()
    captured = capsys.readouterr()
    assert "Not enough history" in captured.out

def test_evaluate_health_normal(mock_env, capsys):
    rhr, out_dir = mock_env
    
    # Need at least 60 days
    dates = pd.date_range("2020-01-01", periods=65, freq="D")
    
    # Construct normal data
    df = pd.DataFrame({
        "Date": dates,
        "Exec_Slippage_Mean": [0.001] * 65, # Very stable slippage
        "Delay_Cost_Mean": [0.0005] * 65,
        "Idiosyncratic_Alpha": [0.05] * 65, # Positive alpha, stable
        "Exposure_Size": [0.0] * 65, # Normal size
        "Exposure_Momentum": [0.0] * 65,
        "Exposure_Volatility": [0.0] * 65,
        "Win_Rate": [0.55] * 65,
        "Payoff_Ratio": [1.5] * 65,
        "Max_DD": [-0.05] * 65,
        "Calmar": [2.0] * 65,
        "Sharpe": [2.0] * 65
    })
    
    df.to_csv(out_dir / "rolling_metrics_20.csv", index=False)
    df.to_csv(out_dir / "rolling_metrics_60.csv", index=False)
    
    rhr.evaluate_health()
    
    # If the report is generated, it prints markdown but maybe no alerts
    captured = capsys.readouterr()
    assert "Rolling Health Summary" in captured.out
    
def test_evaluate_health_anomaly(mock_env, capsys):
    rhr, out_dir = mock_env
    
    # Construct data with an anomaly at the very end
    dates = pd.date_range("2020-01-01", periods=65, freq="D")
    
    slip = [0.001] * 64 + [-0.05] # Sudden huge negative slippage at the end
    delay = [0.0005] * 64 + [-0.05] # Sudden huge delay cost
    
    # Alpha drops below zero
    alpha_20 = [0.05] * 60 + [-0.01] * 5
    alpha_60 = [0.05] * 65
    
    # Size exposure goes to minimum
    size = [0.0] * 64 + [-5.0]
    
    df_20 = pd.DataFrame({
        "Date": dates,
        "Exec_Slippage_Mean": slip,
        "Delay_Cost_Mean": delay,
        "Idiosyncratic_Alpha": alpha_20,
        "Exposure_Size": size,
        "Exposure_Momentum": [0.0] * 65,
        "Exposure_Volatility": [0.0] * 65,
        "Win_Rate": [0.55] * 65,
        "Payoff_Ratio": [1.5] * 65,
        "Max_DD": [-0.05] * 65,
        "Calmar": [2.0] * 65,
        "Sharpe": [2.0] * 65
    })
    
    df_60 = df_20.copy()
    df_60["Idiosyncratic_Alpha"] = alpha_60
    
    df_20.to_csv(out_dir / "rolling_metrics_20.csv", index=False)
    df_60.to_csv(out_dir / "rolling_metrics_60.csv", index=False)
    
    rhr.evaluate_health()
    
    captured = capsys.readouterr()
    out = captured.out
    assert "Rolling Health Summary" in out
    
    # We should have all the alerts we triggered
    assert "执行摩擦崩盘" in out
    assert "隔夜跳空恶化" in out
    assert "Alpha Decay (阿尔法衰减)" in out
    assert "极端微盘暴露" in out
