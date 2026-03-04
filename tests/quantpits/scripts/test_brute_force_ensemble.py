import pytest
import pandas as pd
import numpy as np
import yaml
from unittest.mock import MagicMock

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, brute_force_ensemble
    import importlib
    importlib.reload(env)
    importlib.reload(brute_force_ensemble)
    
    yield brute_force_ensemble, workspace

# ── zscore_norm ──────────────────────────────────────────────────────────
def test_zscore_norm(mock_env):
    bfe, _ = mock_env
    series = pd.Series([1, 2, 3, 10, 20, 30], 
                       index=pd.MultiIndex.from_arrays([
                           pd.to_datetime(["2020-01-01"]*3 + ["2020-01-02"]*3),
                           ["A", "B", "C", "A", "B", "C"]
                       ], names=["datetime", "instrument"]))
    normed = bfe.zscore_norm(series)
    assert np.isclose(normed.xs("2020-01-01", level="datetime").mean(), 0)
    assert np.isclose(normed.xs("2020-01-01", level="datetime").std(), 1)

# ── load_combo_groups ────────────────────────────────────────────────────
def test_load_combo_groups(mock_env):
    bfe, workspace = mock_env
    cfg_path = workspace / "combo_groups.yaml"
    cfg_data = {
        "groups": {
            "G1": ["m1", "m2", "invalid_m"],
            "G2": ["m3"]
        }
    }
    cfg_path.write_text(yaml.dump(cfg_data))
    
    groups = bfe.load_combo_groups(str(cfg_path), available_models=["m1", "m2", "m3", "m4"])
    assert "G1" in groups
    assert "G2" in groups
    assert groups["G1"] == ["m1", "m2"] # invalid_m is filtered
    assert groups["G2"] == ["m3"]

def test_load_combo_groups_empty(mock_env):
    bfe, workspace = mock_env
    cfg_path = workspace / "combo_groups.yaml"
    cfg_path.write_text(yaml.dump({}))
    
    with pytest.raises(ValueError, match="为空"):
        bfe.load_combo_groups(str(cfg_path), available_models=["m1"])

# ── generate_grouped_combinations ────────────────────────────────────────
def test_generate_grouped_combinations(mock_env):
    bfe, _ = mock_env
    groups = {
        "G1": ["m1", "m2"],
        "G2": ["m3"],
        "G3": ["m4"]
    }
    # Without min_combo_size/max_combo_size args (default 1, 0)
    combos = bfe.generate_grouped_combinations(groups)
    # Total combos should cover r=1 to r=3.
    # r=1: (m1,), (m2,), (m3,), (m4,) => 4
    # r=2: G1*G2 => 2, G1*G3 => 2, G2*G3 => 1 => 5
    # r=3: G1*G2*G3 => 2
    # Total: 11
    assert len(combos) == 11
    assert ("m1", "m3") in combos

def test_generate_grouped_combinations_min_max(mock_env):
    bfe, _ = mock_env
    groups = {
        "G1": ["m1", "m2"],
        "G2": ["m3"],
        "G3": ["m4"]
    }
    combos = bfe.generate_grouped_combinations(groups, min_combo_size=2, max_combo_size=2)
    # r=2 only: 5 combos
    assert len(combos) == 5
    assert ("m1",) not in combos
    assert ("m1", "m3", "m4") not in combos
    assert ("m2", "m4") in combos

# ── split_is_oos_by_args ─────────────────────────────────────────────────
def test_split_is_oos_by_args(mock_env):
    bfe, _ = mock_env
    
    dates = pd.date_range("2010-01-01", "2020-01-01", freq="YS") # 11 years
    df = pd.DataFrame({"score": range(len(dates))}, index=pd.MultiIndex.from_arrays([
        dates, ["A"]*len(dates)
    ], names=["datetime", "instrument"]))
    
    args = MagicMock()
    args.start_date = None
    args.end_date = None
    args.exclude_last_years = 2
    args.exclude_last_months = 0
    
    is_df, oos_df = bfe.split_is_oos_by_args(df, args)
    # Max date is 2020-01-01. cutoff is 2018-01-01.
    is_dates = is_df.index.get_level_values("datetime").unique()
    oos_dates = oos_df.index.get_level_values("datetime").unique()
    print(is_dates)
    print(oos_dates)
    
    assert "2018-01-01" in is_dates
    assert "2019-01-01" not in is_dates
    
    assert "2019-01-01" in oos_dates
    assert "2020-01-01" in oos_dates
    assert "2018-01-01" not in oos_dates

# ── extract_report_df ────────────────────────────────────────────────────
def test_extract_report_df(mock_env):
    bfe, _ = mock_env
    df = pd.DataFrame({"A": [1]})
    
    metrics_dict = {"fold1": (df, "other")}
    assert bfe.extract_report_df(metrics_dict).equals(df)
    
    metrics_tuple = (df, "other")
    assert bfe.extract_report_df(metrics_tuple).equals(df)
    
    assert bfe.extract_report_df(df).equals(df)
