import os
import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Apply environment mocking before loading the module
@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    config_dir = workspace / "config"
    config_dir.mkdir()
    
    output_dir = workspace / "output"
    output_dir.mkdir()
    
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, signal_ranking
    import importlib
    importlib.reload(env)
    
    # Reload signal_ranking to pick up the new env.ROOT_DIR if needed
    importlib.reload(signal_ranking)
    
    # Since signal_ranking defines its constants at module level during import,
    # we need to patch them directly to use our tmp_path.
    monkeypatch.setattr(signal_ranking, 'ROOT_DIR', str(workspace))
    monkeypatch.setattr(signal_ranking, 'PREDICTION_DIR', str(pred_dir))
    monkeypatch.setattr(signal_ranking, 'ENSEMBLE_CONFIG_FILE', str(config_dir / "ensemble_config.json"))
    
    yield signal_ranking, workspace

# ── parse_ensemble_config ────────────────────────────────────────────────
def test_parse_ensemble_config_new_format(mock_env):
    sr, workspace = mock_env
    config_file = workspace / "config" / "ensemble_config.json"
    
    cfg = {
        "combos": {
            "combo_A": {"models": ["m1"], "method": "equal", "default": True}
        }
    }
    config_file.write_text(json.dumps(cfg))
    
    # signal_ranking.parse_ensemble_config takes no arguments, reads from ENSEMBLE_CONFIG_FILE
    result = sr.parse_ensemble_config()
    assert "combo_A" in result
    assert result["combo_A"]["method"] == "equal"

def test_parse_ensemble_config_old_format(mock_env):
    sr, workspace = mock_env
    config_file = workspace / "config" / "ensemble_config.json"
    
    cfg = {
        "models": ["m1", "m2"],
        "ensemble_method": "icir_weighted"
    }
    config_file.write_text(json.dumps(cfg))
    
    result = sr.parse_ensemble_config()
    assert "legacy" in result
    assert result["legacy"]["method"] == "icir_weighted"
    assert result["legacy"]["models"] == ["m1", "m2"]

def test_parse_ensemble_config_empty(mock_env):
    sr, workspace = mock_env
    result = sr.parse_ensemble_config()
    assert result == {}

# ── get_default_combo ────────────────────────────────────────────────────
def test_get_default_combo(mock_env):
    sr, _ = mock_env
    combos = {
        "combo_1": {"default": False},
        "combo_2": {"default": True},
        "combo_3": {"default": False}
    }
    name, cfg = sr.get_default_combo(combos)
    assert name == "combo_2"
    assert cfg["default"] is True

def test_get_default_combo_no_default(mock_env):
    sr, _ = mock_env
    combos = {
        "combo_1": {"default": False},
        "combo_2": {"default": False}
    }
    name, cfg = sr.get_default_combo(combos)
    assert name == "combo_1"

# ── generate_signal_scores ───────────────────────────────────────────────
def test_generate_signal_scores(mock_env):
    sr, _ = mock_env
    
    df = pd.DataFrame({
        "score": [0.1, 0.5, 0.9, 0.2]
    }, index=pd.MultiIndex.from_tuples([
        ("000001", "2026-03-01"),
        ("000002", "2026-03-01"),
        ("000003", "2026-03-01"),
        ("000004", "2026-03-01")
    ], names=["instrument", "datetime"]))
    
    # Raw range is [0.1, 0.9].
    # Normalizing to [-100, 100]:
    # 0.1 -> -100
    # 0.5 -> 0
    # 0.9 -> 100
    res_df, latest_date = sr.generate_signal_scores(df, top_n=3)
    
    assert latest_date == "2026-03-01"
    assert len(res_df) == 3
    assert list(res_df.index) == ["000003", "000002", "000004"]
    assert res_df.loc["000003", "推荐指数"] == 100.0
    assert res_df.loc["000002", "推荐指数"] == 0.0
    assert res_df.loc["000004", "推荐指数"] == -75.0

def test_generate_signal_scores_constant(mock_env):
    sr, _ = mock_env
    
    df = pd.DataFrame({
        "score": [0.5, 0.5, 0.5]
    }, index=pd.MultiIndex.from_tuples([
        ("000001", "2026-03-01"),
        ("000002", "2026-03-01"),
        ("000003", "2026-03-01"),
    ], names=["instrument", "datetime"]))
    
    res_df, latest_date = sr.generate_signal_scores(df, top_n=2)
    assert len(res_df) == 2
    assert res_df["推荐指数"].iloc[0] == 0.0

# ── find_prediction_file ─────────────────────────────────────────────────
def test_find_prediction_file(mock_env):
    sr, workspace = mock_env
    pred_dir = workspace / "output" / "predictions"
    
    # Create some mock files
    (pred_dir / "ensemble_2026-03-01.csv").write_text("1")
    (pred_dir / "ensemble_2026-03-02.csv").write_text("2")
    (pred_dir / "ensemble_combo_A_2026-03-01.csv").write_text("3")
    
    # Generic latest
    res = sr.find_prediction_file(combo_name=None)
    assert "ensemble_2026-03-02.csv" in res
    
    # Specific combo
    res = sr.find_prediction_file(combo_name="combo_A")
    assert "ensemble_combo_A_2026-03-01.csv" in res

def test_find_prediction_file_not_found(mock_env):
    sr, _ = mock_env
    
    with pytest.raises(FileNotFoundError):
        sr.find_prediction_file(combo_name="combo_X")
        
    with pytest.raises(FileNotFoundError):
        sr.find_prediction_file(combo_name=None)
