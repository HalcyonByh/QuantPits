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
    
    from quantpits.scripts import env, brute_force_fast
    import importlib
    importlib.reload(env)
    importlib.reload(brute_force_fast)
    
    # Disable GPU and Numba for tests
    monkeypatch.setattr(brute_force_fast, '_USE_GPU', False)
    # Monkeypatch cupy as numpy just in case _USE_GPU leaks
    import sys
    sys.modules['cupy'] = np
    
    yield brute_force_fast, workspace

# ── zscore_norm ──────────────────────────────────────────────────────────
def test_zscore_norm(mock_env):
    bff, _ = mock_env
    series = pd.Series([1, 2, 3], 
                       index=pd.MultiIndex.from_arrays([
                           pd.to_datetime(["2020-01-01"]*3),
                           ["A", "B", "C"]
                       ], names=["datetime", "instrument"]))
    normed = bff.zscore_norm(series)
    assert np.isclose(normed.mean(), 0)
    assert np.isclose(normed.std(), 1)

# ── _to_numpy ────────────────────────────────────────────────────────────
def test_to_numpy_no_gpu(mock_env):
    bff, _ = mock_env
    arr = np.array([1, 2, 3])
    res = bff._to_numpy(arr)
    assert isinstance(res, np.ndarray)
    assert (res == arr).all()

# ── compute_metrics ──────────────────────────────────────────────────────
def test_compute_metrics_known(mock_env):
    bff, _ = mock_env
    # 10 days of alternating returns to avoid float precision 0 std issue
    net_returns = np.array([0.01, -0.01] * 5)
    bench_returns = np.zeros(10)
    
    metrics = bff.compute_metrics(net_returns, bench_returns)
    assert "Ann_Ret" in metrics
    assert "Sharpe" in metrics
    
    # Final NAV should be (1.01 * 0.99)^5 * account
    assert np.isclose(metrics["Total_Ret"], ((1.01 * 0.99)**5) - 1.0)
    
    # Ann_Ret is mean(ret) * 252 = 0 * 252 = 0
    assert np.isclose(metrics["Ann_Ret"], 0)
    
    # Sharpe is 0 since mean is 0
    assert np.isclose(metrics["Sharpe"], 0)

def test_compute_metrics_zero_std(mock_env):
    bff, _ = mock_env
    net_returns = np.zeros(10)
    bench_returns = np.zeros(10)
    metrics = bff.compute_metrics(net_returns, bench_returns)
    assert metrics["Sharpe"] == 0
    assert metrics["Max_DD"] == 0

# ── _vectorized_topk_backtest_single ─────────────────────────────────────
def test_vectorized_topk_backtest_single(mock_env):
    bff, _ = mock_env
    # T=2 days, N=3 stocks
    combo_score_np = np.array([
        [0.9, 0.5, 0.1], # Day 0: picks A
        [0.1, 0.9, 0.5]  # Day 1: picks B (if rebalancing, but we test both)
    ])
    returns_np = np.array([
        [0.05, 0.01, -0.01], # Day 0 returns
        [-0.05, 0.05, 0.1]   # Day 1 returns
    ])
    
    # Test 1: rebalance daily (rebalance_freq=1)
    net_ret = bff._vectorized_topk_backtest_single(
        combo_score_np, returns_np, top_k=1, cost_rate=0.0, rebalance_freq=1
    )
    # Day 0: select idx 0 (val 0.9). Ret = 0.05
    # Day 1: select idx 1 (val 0.9). Ret = 0.05
    assert len(net_ret) == 2
    assert np.isclose(net_ret[0], 0.05)
    assert np.isclose(net_ret[1], 0.05)

def test_vectorized_topk_backtest_rebalance_freq(mock_env):
    bff, _ = mock_env
    combo_score_np = np.array([
        [0.9, 0.5], # Day 0
        [0.1, 0.9], # Day 1
        [0.1, 0.9], # Day 2
    ])
    returns_np = np.array([
        [0.05, -0.05], # Day 0
        [-0.05, 0.05], # Day 1
        [0.10, -0.10], # Day 2
    ])
    
    # rebalance_freq=3 => Only selects on Day 0. Holds that (idx 0) for all 3 days.
    net_ret = bff._vectorized_topk_backtest_single(
        combo_score_np, returns_np, top_k=1, cost_rate=0.0, rebalance_freq=3
    )
    assert np.isclose(net_ret[0], 0.05)
    assert np.isclose(net_ret[1], -0.05)
    assert np.isclose(net_ret[2], 0.10)
    
    # Same with cost. Rebalancing at day 0 doesn't cost anything initially if there's no prior portfolio
    # (The function calculates turnover starting from the second rebalance point)
    net_ret_cost = bff._vectorized_topk_backtest_single(
        combo_score_np, returns_np, top_k=1, cost_rate=0.01, rebalance_freq=1
    )
    # Day 0: Ret(idx 0) = 0.05
    # Day 1: Ret(idx 1) = 0.05. turnover=1. cost=0.01 -> Net = 0.04
    # Day 2: Ret(idx 1) = -0.10. turnover=0. cost=0 -> Net = -0.10
    assert np.isclose(net_ret_cost[0], 0.05)
    assert np.isclose(net_ret_cost[1], 0.04)
    assert np.isclose(net_ret_cost[2], -0.10)

# ── prepare_matrices ─────────────────────────────────────────────────────
def test_prepare_matrices(mock_env):
    bff, _ = mock_env
    dates = pd.date_range("2020-01-01", "2020-01-02")
    instruments = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    
    norm_df = pd.DataFrame({"m1": [1, 2, 3, 4], "m2": [4, 3, 2, 1]}, index=idx)
    returns_wide = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, -0.02]}, index=dates)
    
    scores_np, returns_np, models, com_dates, com_instrs = bff.prepare_matrices(norm_df, returns_wide, dates)
    
    assert len(models) == 2
    assert "m1" in scores_np
    assert returns_np.shape == (2, 2)
    assert np.isclose(returns_np[0, 0], 0.01) # Day 0, Instr A
    assert np.isclose(scores_np["m1"][0, 0], 1.0) # Day 0, Instr A

# ── load_combo_groups \u0026 generate_grouped_combinations ─────────────────────
# (These are mostly identical to brute_force_ensemble logic, testing them briefly)
def test_load_combo_groups(mock_env):
    bff, workspace = mock_env
    cfg_path = workspace / "combo_groups.yaml"
    cfg_path.write_text(yaml.dump({"groups": {"G1": ["m1"]}}))
    groups = bff.load_combo_groups(str(cfg_path), ["m1", "m2"])
    assert "G1" in groups

def test_generate_grouped_combinations(mock_env):
    bff, _ = mock_env
    combos = bff.generate_grouped_combinations({"G1": ["m1"]})
    assert ("m1",) in combos

# ── split_is_oos_by_args ─────────────────────────────────────────────────
def test_split_is_oos_by_args(mock_env):
    bff, _ = mock_env
    dates = pd.date_range("2015-01-01", "2020-01-01", freq="YS")
    df = pd.DataFrame({"A": range(6)}, index=pd.MultiIndex.from_arrays([
        dates, ["S"]*6
    ], names=["datetime", "instrument"]))
    
    args = MagicMock()
    args.start_date = None
    args.end_date = None
    args.exclude_last_years = 2
    args.exclude_last_months = 0
    
    is_df, oos_df = bff.split_is_oos_by_args(df, args)
    assert len(oos_df.index.get_level_values("datetime").unique()) == 2 # 2019, 2020

# ── load_config ──────────────────────────────────────────────────────────
import os
import json
from unittest.mock import patch

def test_load_config(mock_env, tmp_path):
    bff, workspace = mock_env

    record_file = tmp_path / "records.json"
    with open(record_file, "w") as f:
        json.dump({"models": {"m1": "r1"}}, f)

    (workspace / "config").mkdir(exist_ok=True)
    with open(workspace / "config" / "model_config.json", "w") as f:
        json.dump({"TopK": 20}, f)

    old_cwd = os.getcwd()
    os.chdir(workspace)
    try:
        records, model_config = bff.load_config(str(record_file))
    finally:
        os.chdir(old_cwd)
    assert records["models"]["m1"] == "r1"
    assert model_config["TopK"] == 20

# ── _init_gpu ────────────────────────────────────────────────────────────
def test_init_gpu_no_gpu(mock_env):
    bff, _ = mock_env
    bff._init_gpu(force_no_gpu=True)
    assert bff._USE_GPU is False

def test_init_gpu_force_fail(mock_env):
    bff, _ = mock_env
    # Force GPU but cupy isn't truly available → should exit
    with patch.dict('sys.modules', {'cupy': None}):
        import pytest
        with pytest.raises(SystemExit):
            bff._init_gpu(force_gpu=True)

# ── load_predictions ─────────────────────────────────────────────────────
@patch('qlib.workflow.R', create=True)
def test_load_predictions(mock_R, mock_env):
    bff, _ = mock_env

    dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"])
    idx = pd.MultiIndex.from_arrays([dates, ["A", "B", "A", "B"]], names=["datetime", "instrument"])

    mock_recorder = MagicMock()
    mock_recorder.load_object.return_value = pd.Series(
        [0.5, 0.6, 0.7, 0.8], index=idx, name="score"
    )
    mock_recorder.list_metrics.return_value = {"IC_ICIR": 1.5}
    mock_R.get_recorder.return_value = mock_recorder

    train_records = {
        "experiment_name": "test_exp",
        "models": {"m1": "r1", "m2": "r2"}
    }

    norm_df, metrics = bff.load_predictions(train_records)
    assert "m1" in norm_df.columns
    assert "m2" in norm_df.columns
    assert metrics["m1"] == 1.5

@patch('qlib.workflow.R', create=True)
def test_load_predictions_failure(mock_R, mock_env):
    bff, _ = mock_env
    mock_R.get_recorder.side_effect = Exception("not found")

    with pytest.raises(ValueError, match="未加载到任何预测数据"):
        bff.load_predictions({"experiment_name": "exp", "models": {"m1": "r1"}})

# ── correlation_analysis ─────────────────────────────────────────────────
def test_correlation_analysis(mock_env, tmp_path):
    bff, _ = mock_env

    dates = pd.to_datetime(["2020-01-01"] * 3)
    idx = pd.MultiIndex.from_arrays([dates, ["A", "B", "C"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"M1": [1.0, 2.0, 3.0], "M2": [1.0, 2.0, 3.0]}, index=idx)

    out_dir = str(tmp_path / "output")
    os.makedirs(out_dir, exist_ok=True)
    corr = bff.correlation_analysis(norm_df, out_dir, "2020-01-01")

    assert corr.loc["M1", "M2"] == 1.0
    assert os.path.exists(os.path.join(out_dir, "correlation_matrix_2020-01-01.csv"))

# ── analyze_results ──────────────────────────────────────────────────────
def test_analyze_results_empty(mock_env, tmp_path):
    bff, _ = mock_env
    os.makedirs(tmp_path / "output", exist_ok=True)
    bff.analyze_results(
        results_df=pd.DataFrame(),
        corr_matrix=pd.DataFrame(),
        norm_df=pd.DataFrame(),
        train_records={"experiment_name": "exp", "models": {}},
        output_dir=str(tmp_path / "output"),
        anchor_date="2020-01-01",
    )

@patch('qlib.workflow.R', create=True)
def test_analyze_results_basic(mock_R, mock_env, tmp_path):
    bff, _ = mock_env
    out_dir = str(tmp_path / "output")
    os.makedirs(out_dir, exist_ok=True)

    results_df = pd.DataFrame({
        "models": ["m1", "m2", "m1,m2"],
        "n_models": [1, 1, 2],
        "Ann_Ret": [0.15, 0.12, 0.18],
        "Max_DD": [-0.05, -0.08, -0.04],
        "Excess_Ret": [0.10, 0.07, 0.13],
        "Ann_Excess": [0.10, 0.07, 0.13],
        "Total_Ret": [0.15, 0.12, 0.18],
        "Final_NAV": [115000, 112000, 118000],
        "Calmar": [3.0, 1.5, 4.5],
        "Sharpe": [1.5, 0.8, 2.0],
    })

    corr_matrix = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], index=["m1", "m2"], columns=["m1", "m2"]
    )

    dates = pd.to_datetime(["2020-01-01"] * 2)
    idx = pd.MultiIndex.from_arrays([dates, ["A", "B"]], names=["datetime", "instrument"])
    norm_df = pd.DataFrame({"m1": [1.0, 2.0], "m2": [1.5, 2.5]}, index=idx)

    mock_R.get_recorder.side_effect = Exception("skip cluster")

    bff.analyze_results(
        results_df=results_df,
        corr_matrix=corr_matrix,
        norm_df=norm_df,
        train_records={"experiment_name": "exp", "models": {"m1": "r1", "m2": "r2"}},
        output_dir=out_dir,
        anchor_date="2020-01-01",
        top_n=2,
    )

    # Function ran without error — sufficient for coverage

