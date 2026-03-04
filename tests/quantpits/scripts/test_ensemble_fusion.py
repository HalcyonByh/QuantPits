import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, ensemble_fusion
    import importlib
    importlib.reload(env)
    importlib.reload(ensemble_fusion)
    
    monkeypatch.setattr(ensemble_fusion, 'ROOT_DIR', str(workspace))
    
    yield ensemble_fusion, workspace

# ── parse_ensemble_config ────────────────────────────────────────────────
def test_parse_ensemble_config_new(mock_env):
    ef, _ = mock_env
    cfg = {
        "min_model_ic": 0.05,
        "combos": {
            "A": {"method": "equal", "models": ["m1", "m2"]}
        }
    }
    combos, global_cfg = ef.parse_ensemble_config(cfg)
    assert "A" in combos
    assert global_cfg["min_model_ic"] == 0.05

def test_parse_ensemble_config_old(mock_env):
    ef, _ = mock_env
    cfg = {
        "models": ["m1", "m2"],
        "ensemble_method": "manual",
        "manual_weights": {"m1": 0.8, "m2": 0.2},
        "min_model_ic": 0.01
    }
    combos, global_cfg = ef.parse_ensemble_config(cfg)
    assert "legacy" in combos
    assert combos["legacy"]["method"] == "manual"
    assert combos["legacy"]["manual_weights"]["m1"] == 0.8
    assert global_cfg["min_model_ic"] == 0.01

def test_parse_ensemble_config_empty(mock_env):
    ef, _ = mock_env
    combos, global_cfg = ef.parse_ensemble_config({})
    assert combos == {}
    assert global_cfg == {}

# ── get_default_combo ────────────────────────────────────────────────────
def test_get_default_combo(mock_env):
    ef, _ = mock_env
    combos = {
        "c1": {"default": False},
        "c2": {"default": True}
    }
    name, cfg = ef.get_default_combo(combos)
    assert name == "c2"

# ── zscore_norm ──────────────────────────────────────────────────────────
def test_zscore_norm(mock_env):
    ef, _ = mock_env
    series = pd.Series([1, 2, 3, 10, 20, 30], 
                       index=pd.MultiIndex.from_arrays([
                           pd.to_datetime(["2020-01-01"]*3 + ["2020-01-02"]*3),
                           ["A", "B", "C", "A", "B", "C"]
                       ], names=["datetime", "instrument"]))
    normed = ef.zscore_norm(series)
    # Mean of 1,2,3 is 2. std is ~1.
    assert np.isclose(normed.xs("2020-01-01", level="datetime").mean(), 0)
    assert np.isclose(normed.xs("2020-01-01", level="datetime").std(), 1)
    assert np.isclose(normed.xs("2020-01-02", level="datetime").mean(), 0)

def test_zscore_norm_zero_std(mock_env):
    ef, _ = mock_env
    series = pd.Series([5, 5, 5], 
                       index=pd.MultiIndex.from_arrays([
                           pd.to_datetime(["2020-01-01"]*3),
                           ["A", "B", "C"]
                       ], names=["datetime", "instrument"]))
    normed = ef.zscore_norm(series)
    assert (normed == 0).all()

# ── calculate_weights ────────────────────────────────────────────────────
def _make_norm_df():
    return pd.DataFrame({"m1": [1, 2], "m2": [3, 4]}).rename_axis("datetime")

def test_calculate_weights_equal(mock_env):
    ef, _ = mock_env
    df = _make_norm_df()
    dyn_w, stat_w, is_dyn = ef.calculate_weights(df, {}, "equal", {}, {})
    assert not is_dyn
    assert stat_w == {"m1": 0.5, "m2": 0.5}

def test_calculate_weights_icir(mock_env):
    ef, _ = mock_env
    df = _make_norm_df()
    metrics = {"m1": 0.1, "m2": 0.3}
    ens_cfg = {"min_model_ic": 0.05}
    dyn_w, stat_w, is_dyn = ef.calculate_weights(df, metrics, "icir_weighted", {}, ens_cfg)
    assert not is_dyn
    assert stat_w["m1"] == pytest.approx(0.25) # 0.1 / 0.4
    assert stat_w["m2"] == pytest.approx(0.75) # 0.3 / 0.4

def test_calculate_weights_manual(mock_env):
    ef, _ = mock_env
    df = _make_norm_df()
    # Test string parse
    dyn_w, stat_w, is_dyn = ef.calculate_weights(df, {}, "manual", {}, {}, manual_weights_str="m1:0.6, m2:0.4")
    assert not is_dyn
    assert stat_w["m1"] == 0.6
    assert stat_w["m2"] == 0.4

# ── generate_ensemble_signal ─────────────────────────────────────────────
def test_generate_ensemble_signal_static(mock_env):
    ef, _ = mock_env
    df = _make_norm_df()
    weights = {"m1": 0.8, "m2": 0.2}
    signal = ef.generate_ensemble_signal(df, None, weights, False)
    assert signal.iloc[0] == 1*0.8 + 3*0.2
    assert signal.iloc[1] == 2*0.8 + 4*0.2

# ── filter_norm_df_by_args ───────────────────────────────────────────────
def test_filter_norm_df_by_args(mock_env):
    ef, _ = mock_env
    dates = pd.date_range("2020-01-01", "2020-01-10")
    df = pd.DataFrame({"score": range(10)}, index=pd.MultiIndex.from_arrays([
        dates, ["A"]*10
    ], names=["datetime", "instrument"]))
    
    args = MagicMock()
    args.start_date = "2020-01-03"
    args.end_date = "2020-01-05"
    args.only_last_years = 0
    args.only_last_months = 0
    
    filtered = ef.filter_norm_df_by_args(df, args)
    assert len(filtered) == 3
    
    # Test only_last_months (1 month is larger than 10 days, will keep all)
    args2 = MagicMock()
    args2.start_date = None
    args2.end_date = None
    args2.only_last_years = 0
    args2.only_last_months = 1
    
    # Needs to construct df over 2 months
    dates2 = pd.date_range("2020-01-01", "2020-03-01", freq="MS")
    df2 = pd.DataFrame({"score": range(len(dates2))}, index=pd.MultiIndex.from_arrays([
        dates2, ["A"]*len(dates2)
    ], names=["datetime", "instrument"]))
    filtered2 = ef.filter_norm_df_by_args(df2, args2)
    # 2020-03-01 is max. -1 month is 2020-02-01. strictly > 2020-02-01 means only 2020-03-01 is kept.
    assert len(filtered2) == 1

# ── calculate_safe_risk ──────────────────────────────────────────────────
@patch('qlib.contrib.evaluate.risk_analysis')
def test_calculate_safe_risk(mock_risk, mock_env):
    ef, _ = mock_env
    # mock returns a series
    mock_risk.return_value = pd.Series({"Ann_Ret": 0.1, "Max_DD": -0.05})
    res = ef.calculate_safe_risk(pd.Series([0.01, 0.02]), "day")
    assert res == {"Ann_Ret": 0.1, "Max_DD": -0.05}

# ── extract_report_df ────────────────────────────────────────────────────
def test_extract_report_df_dict(mock_env):
    ef, _ = mock_env
    df = pd.DataFrame({"A": [1]})
    metrics = {"key": (df, "other")}
    assert ef.extract_report_df(metrics).equals(df)

def test_extract_report_df_tuple(mock_env):
    ef, _ = mock_env
    df = pd.DataFrame({"A": [1]})
    metrics = (df, "other")
    assert ef.extract_report_df(metrics).equals(df)
