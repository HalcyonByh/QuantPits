import os
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_env_constants(tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    with patch.dict(os.environ, {"QLIB_WORKSPACE_DIR": str(workspace)}):
        import quantpits.scripts.env as env
        with patch('quantpits.scripts.env.ROOT_DIR', str(workspace)):
            import quantpits.scripts.ensemble_fusion as ef
            yield ef, workspace

def test_parse_ensemble_config():
    import quantpits.scripts.ensemble_fusion as ef
    
    legacy_config = {
        "models": ["A", "B"],
        "ensemble_method": "manual",
        "manual_weights": {"A": 0.6, "B": 0.4},
        "min_model_ic": 0.05
    }
    
    combos, global_cfg = ef.parse_ensemble_config(legacy_config)
    
    assert "legacy" in combos
    assert combos["legacy"]["method"] == "manual"
    assert combos["legacy"]["models"] == ["A", "B"]
    assert combos["legacy"]["manual_weights"]["A"] == 0.6
    assert global_cfg["min_model_ic"] == 0.05
    
    new_config = {
        "min_model_ic": 0.02,
        "combos": {
            "combo_X": {"models": ["C", "D"], "method": "equal", "default": True}
        }
    }
    
    combos2, global_cfg2 = ef.parse_ensemble_config(new_config)
    assert "combo_X" in combos2
    assert combos2["combo_X"]["default"] is True
    assert global_cfg2["min_model_ic"] == 0.02

def test_get_default_combo():
    import quantpits.scripts.ensemble_fusion as ef
    
    combos = {
        "combo1": {"default": False, "models": ["A"]},
        "combo2": {"default": True, "models": ["B"]}
    }
    name, cfg = ef.get_default_combo(combos)
    assert name == "combo2"
    assert cfg["models"] == ["B"]
    
    combos_none = {
        "combo1": {"models": ["A"]},
        "combo2": {"models": ["B"]}
    }
    name2, cfg2 = ef.get_default_combo(combos_none)
    assert name2 == "combo1"

def test_zscore_norm():
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"])
    instruments = ["A", "B", "A", "B"]
    idx = pd.MultiIndex.from_arrays([dates, instruments], names=['datetime', 'instrument'])
    
    series = pd.Series([10.0, 20.0, 5.0, 5.0], index=idx)
    
    norm = ef.zscore_norm(series)
    
    # 2020-01-01 -> mean=15, std=7.07
    # 2020-01-02 -> mean=5, std=0 (so it should return 0)
    assert np.isclose(norm.loc[("2020-01-01", "A")], -0.7071, atol=1e-3)
    assert np.isclose(norm.loc[("2020-01-01", "B")], 0.7071, atol=1e-3)
    
    assert norm.loc[("2020-01-02", "A")] == 0.0
    assert norm.loc[("2020-01-02", "B")] == 0.0

@patch('qlib.workflow.R')
def test_load_selected_predictions(mock_R, mock_env_constants):
    ef, workspace = mock_env_constants
    
    train_records = {
        "experiment_name": "Exp1",
        "models": {"ModelA": "rid_a", "ModelB": "rid_b"}
    }
    selected_models = ["ModelA", "ModelB", "ModelC"] # ModelC omitted intentionally
    
    mock_recorder_a = MagicMock()
    # Need to return a pandas Series with MultiIndex for prediction
    idx_a = pd.MultiIndex.from_product([[pd.Timestamp("2026-03-01")], ["Inst1", "Inst2"]], names=["datetime", "instrument"])
    mock_recorder_a.load_object.return_value = pd.Series([0.1, 0.2], index=idx_a)
    mock_recorder_a.list_metrics.return_value = {"ICIR": 1.5, "IC": 0.02}
    
    mock_recorder_b = MagicMock()
    idx_b = pd.MultiIndex.from_product([[pd.Timestamp("2026-03-01")], ["Inst1", "Inst2"]], names=["datetime", "instrument"])
    mock_recorder_b.load_object.return_value = pd.DataFrame({"score": [0.4, 0.5]}, index=idx_b)
    mock_recorder_b.list_metrics.return_value = {"ICIR": 0.8}
    
    def r_getter(recorder_id, experiment_name):
        if recorder_id == "rid_a": return mock_recorder_a
        return mock_recorder_b
        
    mock_R.get_recorder.side_effect = r_getter
    
    norm_df, metrics, loaded = ef.load_selected_predictions(train_records, selected_models)
    
    assert loaded == ["ModelA", "ModelB"]
    assert len(norm_df.columns) == 2
    assert "ModelA" in norm_df.columns
    assert "ModelB" in norm_df.columns
    assert metrics["ModelA"] == 1.5
    assert metrics["ModelB"] == 0.8
    # Z-scores computed
    assert np.isclose(norm_df.loc[("2026-03-01", "Inst1"), "ModelA"], -0.7071, atol=1e-3)

def test_calculate_weights_equal():
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.to_datetime(["2020-01-01"])
    idx = pd.MultiIndex.from_arrays([dates, ["A"]], names=['datetime', 'instrument'])
    norm_df = pd.DataFrame({"M1": [1.0], "M2": [2.0]}, index=idx)
    
    model_metrics = {"M1": 0.5, "M2": 0.3}
    model_config = {}
    ensemble_config = {}
    
    final, static, is_dyn = ef.calculate_weights(norm_df, model_metrics, 'equal', model_config, ensemble_config)
    assert not is_dyn
    assert final is None
    assert static["M1"] == 0.5
    assert static["M2"] == 0.5

def test_calculate_weights_icir():
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.to_datetime(["2020-01-01"])
    idx = pd.MultiIndex.from_arrays([dates, ["A"]], names=['datetime', 'instrument'])
    norm_df = pd.DataFrame({"M1": [1.0], "M2": [2.0], "M3": [3.0]}, index=idx)
    
    model_metrics = {"M1": 0.6, "M2": 0.4, "M3": -0.1} # M3 should be excluded
    model_config = {}
    ensemble_config = {"min_model_ic": 0.05}
    
    _, static, is_dyn = ef.calculate_weights(norm_df, model_metrics, 'icir_weighted', model_config, ensemble_config)
    
    assert not is_dyn
    assert static["M1"] == 0.6
    assert static["M2"] == 0.4
    assert static.get("M3", 0) == 0

def test_calculate_weights_manual():
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.to_datetime(["2020-01-01"])
    idx = pd.MultiIndex.from_arrays([dates, ["A"]], names=['datetime', 'instrument'])
    norm_df = pd.DataFrame({"M1": [1.0], "M2": [2.0]}, index=idx)
    
    _, static, is_dyn = ef.calculate_weights(norm_df, {}, 'manual', {}, {}, manual_weights_str="M1:0.7,M2:0.3")
    assert not is_dyn
    assert static["M1"] == 0.7
    assert static["M2"] == 0.3

def test_generate_ensemble_signal():
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.to_datetime(["2020-01-01"])
    idx = pd.MultiIndex.from_arrays([dates, ["A"]], names=['datetime', 'instrument'])
    norm_df = pd.DataFrame({"M1": [10.0], "M2": [20.0]}, index=idx)
    
    static_w = {"M1": 0.4, "M2": 0.6}
    
    final = ef.generate_ensemble_signal(norm_df, None, static_w, False)
    
    assert final.loc[("2020-01-01", "A")] == (10.0 * 0.4 + 20.0 * 0.6)

@patch('quantpits.scripts.strategy.load_strategy_config')
@patch('quantpits.scripts.strategy.get_backtest_config')
@patch('quantpits.scripts.strategy.create_backtest_strategy')
@patch('qlib.backtest.executor.SimulatorExecutor')
@patch('qlib.backtest.backtest')
def test_run_backtest(mock_bt, mock_executor, mock_create_strat, mock_get_bt_cfg, mock_load_st_cfg, mock_env_constants):
    ef, workspace = mock_env_constants
    
    # Mock strategy dependencies
    mock_load_st_cfg.return_value = {}
    mock_get_bt_cfg.return_value = {"account": 100000.0, "exchange_kwargs": {}}
    mock_create_strat.return_value = MagicMock()
    mock_executor.return_value = MagicMock()
    
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    idx = pd.MultiIndex.from_product([dates, ["InstA"]], names=['datetime', 'instrument'])
    final_score = pd.Series([1.0, 2.0, 3.0], index=idx)
    
    mock_report = pd.DataFrame({
        "account": [100000, 105000, 110000],
        "return": [0.0, 0.05, 0.047],
        "bench": [1.0, 1.05, 1.08]
    }, index=dates)
    
    mock_bt.return_value = (mock_report, None)
    
    report_df = ef.run_backtest(final_score, top_k=50, drop_n=0, benchmark="SH000300", freq="day")
    
    assert report_df is not None
    assert len(report_df) == 3
    assert "nav" in report_df.columns
    assert report_df.iloc[-1]["nav"] == 110000

def test_save_predictions(mock_env_constants, tmp_path):
    ef, workspace = mock_env_constants
    
    dates = pd.to_datetime(["2020-01-01"])
    idx = pd.MultiIndex.from_product([dates, ["InstA"]], names=['datetime', 'instrument'])
    final_score = pd.Series([1.0], index=idx)
    
    # Instance-level mock to avoid global pandas locks
    mock_df = MagicMock()
    final_score.to_frame = MagicMock(return_value=mock_df)
    
    out_dir = tmp_path / "output" / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with patch('quantpits.scripts.ensemble_fusion.os.makedirs') as mock_makedir:
        pred_file = ef.save_predictions(
            final_score, 
            "2020-01-01", 
            "exp1", 
            "equal", 
            ["M1", "M2"], 
            {"M1": 0.5, "M2": 0.6}, 
            {"M1": 0.5, "M2": 0.5}, 
            False, 
            str(out_dir)
        )
    
    assert "ensemble_2020-01-01.csv" in pred_file
    assert mock_makedir.called
    assert mock_df.to_csv.called
    
    config_file = out_dir / "ensemble_fusion_config_2020-01-01.json"
    assert config_file.exists()

@patch("builtins.open")
@patch("os.path.exists")
def test_load_config(mock_exists, mock_open):
    import quantpits.scripts.ensemble_fusion as ef
    
    mock_exists.return_value = True
    
    with patch("quantpits.scripts.ensemble_fusion.json.load") as mock_json_load:
        mock_json_load.side_effect = [
            {"train": "record"},
            {"model": "config"},
            {"ensemble": "config"}
        ]
        
        tr, mc, ec = ef.load_config("dummy.json")
        
        assert tr == {"train": "record"}
        assert mc == {"model": "config"}
        assert ec == {"ensemble": "config"}
        assert mock_json_load.call_count == 3

def test_filter_norm_df_by_args():
    import quantpits.scripts.ensemble_fusion as ef
    from types import SimpleNamespace
    
    dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2021-01-01"])
    idx = pd.MultiIndex.from_arrays([dates, ["A", "B", "A", "B"]], names=["datetime", "instrument"])
    df = pd.DataFrame({"M1": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    
    args = SimpleNamespace(start_date="2020-02-01", end_date="2020-03-01")
    filtered = ef.filter_norm_df_by_args(df, args)
    assert len(filtered) == 2
    
    args_last_months = SimpleNamespace(start_date=None, end_date=None, only_last_years=0, only_last_months=11)
    filtered2 = ef.filter_norm_df_by_args(df, args_last_months)
    assert len(filtered2) == 2
    assert filtered2.index.get_level_values("datetime").min() == pd.Timestamp("2020-03-01")

def test_correlation_analysis(tmp_path):
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "instrument"])
    df = pd.DataFrame({"M1": [1.0, 2.0], "M2": [2.0, 4.0]}, index=idx)
    
    out_dir = tmp_path / "corr"
    corr = ef.correlation_analysis(df, str(out_dir), "2020-01-01", "test_combo")
    
    assert corr.loc["M1", "M2"] == 1.0
    assert (out_dir / "correlation_matrix_test_combo_2020-01-01.csv").exists()

@patch('qlib.data.D', create=True)
def test_calculate_weights_dynamic(mock_D):
    import quantpits.scripts.ensemble_fusion as ef
    
    dates = pd.date_range("2020-01-01", periods=65, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
    
    norm_df = pd.DataFrame({"M1": np.random.randn(130), "M2": np.random.randn(130)}, index=idx)
    
    label_df = pd.DataFrame({"label": np.random.randn(130)}, index=idx)
    mock_D.features.return_value = label_df
    
    model_config = {"TopK": 1}
    ensemble_config = {}
    
    final, static, is_dyn = ef.calculate_weights(norm_df, {}, "dynamic", model_config, ensemble_config)
    
    assert is_dyn is True
    assert static is None
    assert final.shape == (65, 2)

def test_extract_report_df():
    import quantpits.scripts.ensemble_fusion as ef
    df = pd.DataFrame({'a': [1]})
    
    assert ef.extract_report_df({'m': df}).equals(df)
    assert ef.extract_report_df((df, 'other')).equals(df)
    assert ef.extract_report_df(((df, 'x'), 'y')).equals(df)
    assert ef.extract_report_df(df).equals(df)

@patch('qlib.contrib.evaluate.risk_analysis')
def test_calculate_safe_risk(mock_risk):
    import quantpits.scripts.ensemble_fusion as ef
    
    mock_risk.return_value = pd.DataFrame({'risk': [0.1, 0.2]}, index=['a', 'b'])
    res = ef.calculate_safe_risk(pd.Series([0.01]), 'day')
    assert res == {'a': 0.1, 'b': 0.2}
    
    mock_risk.return_value = pd.Series([0.3, 0.4], index=['c', 'd'])
    res2 = ef.calculate_safe_risk(pd.DataFrame({'ret': [0.01]}), 'week')
    assert res2 == {'c': 0.3, 'd': 0.4}
    
    mock_risk.side_effect = Exception("error")
    res3 = ef.calculate_safe_risk(pd.Series([0.01]), 'day')
    assert res3 == {}

@patch('strategy.get_backtest_config')
@patch('strategy.load_strategy_config')
def test_compare_combos(mock_load_st, mock_get_bt, tmp_path):
    import quantpits.scripts.ensemble_fusion as ef
    
    mock_load_st.return_value = {}
    mock_get_bt.return_value = {'account': 100000.0}
    
    report_df = pd.DataFrame({
        'account': [100000.0, 105000.0],
        'return': [0.0, 0.05],
        'bench': [1.0, 1.02]
    })
    
    combo_results = [{
        'name': 'combo1',
        'models': ['M1', 'M2'],
        'method': 'equal',
        'is_default': True,
        'report_df': report_df
    }]
    
    out_dir = tmp_path / "combos"
    comp_df = ef.compare_combos(combo_results, "2020-01-01", str(out_dir), "day")
    
    assert len(comp_df) == 1
    assert comp_df.iloc[0]['combo'] == 'combo1'
    assert comp_df.iloc[0]['total_return'] == 5.0
    assert (out_dir / "combo_comparison_2020-01-01.csv").exists()
    assert (out_dir / "combo_comparison_2020-01-01.png").exists()
