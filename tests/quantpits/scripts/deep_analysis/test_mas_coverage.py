import pytest
import os
import pandas as pd
import json
import yaml
from unittest.mock import patch, MagicMock

# Set environment variable BEFORE importing any quantpits modules
os.environ["QLIB_WORKSPACE_DIR"] = os.getcwd()

from quantpits.scripts.deep_analysis.coordinator import Coordinator
from quantpits.scripts.deep_analysis.config_ledger import (
    _parse_workflow_config, snapshot_configs, save_snapshot, 
    load_previous_snapshot, diff_snapshots
)

# ---------------------------------------------------------------------------
# Coordinator Tests
# ---------------------------------------------------------------------------

def test_coordinator_no_data(tmp_path):
    # Create empty workspace
    ws = tmp_path / "empty_ws"
    ws.mkdir()
    (ws / "data").mkdir()
    
    coord = Coordinator(str(ws))
    coord.discover()
    
    assert coord._data_start_date is None
    assert coord._data_end_date is None
    assert coord._daily_amount_df.empty
    
    windows = coord.generate_windows()
    assert windows == []

def test_coordinator_run_with_failures(mock_workspace):
    coord = Coordinator(mock_workspace)
    
    # Mock an agent that fails
    fail_agent = MagicMock()
    fail_agent.name = "FailAgent"
    fail_agent.analyze.side_effect = Exception("Analysis failed")
    
    # Mock an agent that succeeds
    success_agent = MagicMock()
    success_agent.name = "SuccessAgent"
    from quantpits.scripts.deep_analysis.base_agent import AgentFindings
    success_agent.analyze.return_value = AgentFindings("SuccessAgent", "full", [], [], {})
    
    # Run only 1 window to be fast
    coord.requested_windows = ['full']
    findings = coord.run([fail_agent, success_agent])
    
    assert len(findings) == 2
    # Check fail agent finding
    fail_findings = next(f for f in findings if f.agent_name == "FailAgent")
    assert 'error' in fail_findings.raw_metrics
    assert fail_findings.raw_metrics['error'] == "Analysis failed"
    
    # Check success agent finding
    success_findings = next(f for f in findings if f.agent_name == "SuccessAgent")
    assert success_findings.agent_name == "SuccessAgent"

def test_coordinator_build_context_empty_dfs(mock_workspace):
    # Remove some files to trigger empty DF logic
    os.remove(os.path.join(mock_workspace, "data/trade_log_full.csv"))
    os.remove(os.path.join(mock_workspace, "data/trade_classification.csv"))
    
    coord = Coordinator(mock_workspace)
    coord.discover()
    
    window = {'label': 'full', 'start_date': '2026-01-01', 'end_date': '2026-04-10'}
    ctx = coord.build_context(window)
    
    assert ctx.trade_log_df.empty
    assert ctx.trade_classification_df.empty
    assert not ctx.daily_amount_df.empty

# ---------------------------------------------------------------------------
# ConfigLedger Tests
# ---------------------------------------------------------------------------

def test_parse_workflow_config_error(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    with open(bad_yaml, "w") as f:
        f.write("invalid: [yaml: structure")
    
    res = _parse_workflow_config(str(bad_yaml))
    assert '_error' in res
    assert res['_path'] == str(bad_yaml)

def test_snapshot_configs_missing_files(mock_workspace):
    # Remove config files
    config_dir = os.path.join(mock_workspace, "config")
    for f in os.listdir(config_dir):
        os.remove(os.path.join(config_dir, f))
        
    snapshot = snapshot_configs(mock_workspace)
    assert snapshot['hyperparams'] == {}
    assert snapshot['ensemble_config'] == {}
    assert snapshot['strategy_config'] == {}

def test_snapshot_lifecycle(mock_workspace):
    # Create snapshot
    snap = snapshot_configs(mock_workspace)
    path = save_snapshot(mock_workspace, snap)
    assert os.path.exists(path)
    
    # Load it back
    loaded = load_previous_snapshot(mock_workspace)
    assert loaded['snapshot_date'] == snap['snapshot_date']
    
    # Load before date
    future_date = "2026-12-31"
    loaded_before = load_previous_snapshot(mock_workspace, before_date=future_date)
    assert loaded_before is not None
    
    past_date = "2020-01-01"
    loaded_past = load_previous_snapshot(mock_workspace, before_date=past_date)
    assert loaded_past is None

def test_diff_snapshots_complex(mock_workspace):
    old_snap = {
        'hyperparams': {
            'model_a': {'lr': 0.01, 'dropout': 0.1},
            'model_b': {'lr': 0.02}
        },
        'ensemble_config': {
            'default_combo': 'combo1',
            'combo_groups': {'combo1': ['model_a']}
        },
        'strategy_config': {'topk': 50}
    }
    
    new_snap = {
        'hyperparams': {
            'model_a': {'lr': 0.01, 'dropout': 0.2}, # Change
            'model_c': {'lr': 0.05} # Added
        },
        'ensemble_config': {
            'default_combo': 'combo2', # Switch
            'combo_groups': {'combo1': ['model_a', 'model_c']} # Mutated
        },
        'strategy_config': {'topk': 60} # Change
    }
    
    changes = diff_snapshots(old_snap, new_snap)
    
    types = [c['type'] for c in changes]
    assert 'hyperparam' in types
    assert 'ensemble' in types
    assert 'ensemble_switch' in types
    assert 'strategy' in types
    
    # Specific checks
    hp_changes = [c for c in changes if c['type'] == 'hyperparam']
    assert any(c['key'] == 'model_a.dropout' and c['new'] == 0.2 for c in hp_changes)
    assert any(c['key'] == 'model_c' and c['change'] == 'added' for c in hp_changes)
    assert any(c['key'] == 'model_b' and c['change'] == 'removed' for c in hp_changes)

    switches = [c for c in changes if c['type'] == 'ensemble_switch']
    assert switches[0]['old'] == 'combo1'
    assert switches[0]['new'] == 'combo2'

# ---------------------------------------------------------------------------
# Synthesizer Tests
# ---------------------------------------------------------------------------

def test_synthesizer_rules(mock_analysis_context):
    from quantpits.scripts.deep_analysis.synthesizer import Synthesizer
    from quantpits.scripts.deep_analysis.base_agent import AgentFindings, Finding
    
    # Mock findings to trigger rules
    # 1. Liquidity drift
    port_findings = AgentFindings("Portfolio Risk", "1y", [], [], {
        'style_exposure': {'Barra_Liquidity_Exp_(High-Low)': -0.3},
        'factor_exposure': {'Annualized_Alpha': -0.05, 'Annualized_Alpha_p': 0.5}
    })
    
    # 2. Substitution hit rate
    exec_findings = AgentFindings("Execution Quality", "full", [
        Finding('warning', 'Execution Quality', 'High substitution bias', 'detail', {})
    ], [], {})
    pred_findings = AgentFindings("Prediction Audit", "full", [
        Finding('warning', 'Prediction Audit', 'low buy suggestion hit rate', 'detail', {})
    ], [], {'buy_hit_rate': {'overall': {'hit_rate': 0.6}}}) # > 0.55 for ensemble value
    
    syn = Synthesizer([port_findings, exec_findings, pred_findings])
    res = syn.synthesize()
    
    cross_titles = [f.title for f in res['cross_findings']]
    assert 'Small-cap drift without selection edge' in cross_titles
    assert 'Top convictions increasingly untradeable' in cross_titles
    assert 'Ensemble fusion value confirmed' in cross_titles

def test_synthesizer_alpha_insignificant(mock_analysis_context):
    from quantpits.scripts.deep_analysis.synthesizer import Synthesizer
    from quantpits.scripts.deep_analysis.base_agent import AgentFindings
    
    # Two windows with insignificant alpha
    f1 = AgentFindings("Portfolio Risk", "1y", [], [], {'factor_exposure': {'Annualized_Alpha_p': 0.2}})
    f2 = AgentFindings("Portfolio Risk", "6m", [], [], {'factor_exposure': {'Annualized_Alpha_p': 0.3}})
    
    syn = Synthesizer([f1, f2])
    res = syn.synthesize()
    assert any('No statistically significant alpha' in f.title for f in res['cross_findings'])

def test_synthesizer_health_status():
    from quantpits.scripts.deep_analysis.synthesizer import Synthesizer
    from quantpits.scripts.deep_analysis.base_agent import AgentFindings, Finding
    
    # Critical status
    f1 = AgentFindings("A1", "W1", [Finding('critical', 'A1', 'C1', 'D', {})], [], {})
    f2 = AgentFindings("A2", "W1", [Finding('critical', 'A2', 'C2', 'D', {})], [], {})
    
    syn = Synthesizer([f1, f2])
    assert "CRITICAL" in syn._compute_health_status()
    
    # Warning status
    f3 = AgentFindings("A3", "W1", [Finding('warning', 'A3', f'W{i}', 'D', {}) for i in range(4)], [], {})
    syn2 = Synthesizer([f3])
    assert "WARNING" in syn2._compute_health_status()

# ---------------------------------------------------------------------------
# Agent Detailed Tests
# ---------------------------------------------------------------------------

def test_ensemble_evolution_agent_changes(mock_analysis_context):
    from quantpits.scripts.deep_analysis.agents.ensemble_eval import EnsembleEvolutionAgent
    agent = EnsembleEvolutionAgent()
    
    # Mock config history snapshots
    history_dir = os.path.join(mock_analysis_context.workspace_root, 'data', 'config_history')
    os.makedirs(history_dir, exist_ok=True)
    
    snap1 = {
        'snapshot_date': '2026-01-01',
        'ensemble_config': {'default_combo': 'c1', 'combo_groups': {'c1': ['m1']}}
    }
    snap2 = {
        'snapshot_date': '2026-02-01',
        'ensemble_config': {'default_combo': 'c2', 'combo_groups': {'c1': ['m1', 'm2']}}
    }
    
    with open(os.path.join(history_dir, 'config_snapshot_2026-01-01.json'), 'w') as f:
        json.dump(snap1, f)
    with open(os.path.join(history_dir, 'config_snapshot_2026-02-01.json'), 'w') as f:
        json.dump(snap2, f)
        
    findings = agent.analyze(mock_analysis_context)
    
    event_types = [e['type'] for e in findings.raw_metrics['change_events']]
    assert 'active_switch' in event_types
    assert 'content_mutation' in event_types

def test_ensemble_evolution_correlation_drift(mock_analysis_context):
    from quantpits.scripts.deep_analysis.agents.ensemble_eval import EnsembleEvolutionAgent
    agent = EnsembleEvolutionAgent()
    
    # Create two correlation matrices with large drift
    ensemble_dir = os.path.join(mock_analysis_context.workspace_root, "output/ensemble")
    c1 = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], columns=['m1', 'm2'], index=['m1', 'm2'])
    c2 = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], columns=['m1', 'm2'], index=['m1', 'm2'])
    
    c1.to_csv(os.path.join(ensemble_dir, "correlation_matrix_2026-01-01.csv"))
    c2.to_csv(os.path.join(ensemble_dir, "correlation_matrix_2026-02-01.csv"))
    
    # Update context files
    mock_analysis_context.correlation_matrix_files = [
        os.path.join(ensemble_dir, "correlation_matrix_2026-01-01.csv"),
        os.path.join(ensemble_dir, "correlation_matrix_2026-02-01.csv")
    ]
    
    findings = agent.analyze(mock_analysis_context)
    assert findings.raw_metrics['correlation_drift']['significant_drift'] is True

def test_prediction_audit_agent_sell_hits(mock_analysis_context):
    from quantpits.scripts.deep_analysis.agents.prediction_audit import PredictionAuditAgent
    agent = PredictionAuditAgent()
    
    # Patch both D and init_qlib
    with patch('qlib.data.D') as mock_d, \
         patch('quantpits.scripts.analysis.utils.init_qlib'):
        
        df_ret = pd.DataFrame(
            {'fwd_return': [0.01, 0.03]},
            index=pd.MultiIndex.from_product([[pd.Timestamp('2026-03-20')], ['600000.SH', '000001.SZ']], names=['datetime', 'instrument'])
        )
        mock_d.features.return_value = df_ret
        
        # Test sell direction
        res = agent._analyze_suggestion_hits(mock_analysis_context, direction='sell')
        assert 'overall' in res, f"Expected 'overall' in results, got {res.keys()}"
        assert res['overall']['n_suggestions'] == 2
        assert res['overall']['hit_rate'] == 0.5

def test_prediction_audit_agent_consensus(mock_analysis_context):
    from quantpits.scripts.deep_analysis.agents.prediction_audit import PredictionAuditAgent
    agent = PredictionAuditAgent()
    
    with patch('qlib.data.D') as mock_d, \
         patch('quantpits.scripts.analysis.utils.init_qlib'):
        df_ret = pd.DataFrame(
            {'fwd_return': [0.05, -0.02]},
            index=pd.MultiIndex.from_product([[pd.Timestamp('2026-03-20')], ['600000.SH', '000001.SZ']], names=['datetime', 'instrument'])
        )
        mock_d.features.return_value = df_ret
        
        res = agent._analyze_consensus(mock_analysis_context)
        # 3 files * 1 consensus pick each = 3
        assert res.get('n_high_consensus') == 3, f"Expected 3 high consensus, got {res.get('n_high_consensus')}"
        assert res.get('n_high_divergence') == 3

# ---------------------------------------------------------------------------
# Additional Agent Tests for Coverage
# ---------------------------------------------------------------------------

def test_market_regime_agent_edge_cases(mock_analysis_context):
    from quantpits.scripts.deep_analysis.agents.market_regime import MarketRegimeAgent
    agent = MarketRegimeAgent()
    
    # 1. No benchmark data
    ctx_empty = MagicMock()
    ctx_empty.daily_amount_df = pd.DataFrame()
    findings = agent.analyze(ctx_empty)
    assert any('No benchmark data' in f.title for f in findings.findings)
    
    # 2. Insufficient data
    ctx_short = MagicMock()
    ctx_short.daily_amount_df = pd.DataFrame({'成交日期': pd.date_range('2026-01-01', periods=10), 'CSI300': range(10)})
    findings = agent.analyze(ctx_short)
    assert any('Insufficient data' in f.title for f in findings.findings)
    
    # 3. High volatility and Drawdown
    dates = pd.date_range('2026-01-01', periods=100)
    # First 60 days: low vol. Last 40 days: high vol.
    prices = [100]
    for i in range(59):
        prices.append(prices[-1] * 1.001) # Very low vol
    for i in range(40):
        prices.append(prices[-1] * (0.90 if i % 2 == 0 else 1.10)) # High vol
    
    ctx_vol = MagicMock()
    ctx_vol.daily_amount_df = pd.DataFrame({'成交日期': dates, 'CSI300': prices})
    ctx_vol.window_label = 'full'
    findings = agent.analyze(ctx_vol)
    
    titles = [f.title for f in findings.findings]
    # print(f"DEBUG: Market Regime Titles: {titles}")
    assert any('volatility' in t.lower() for t in titles), f"Volatility finding not found in {titles}"
    assert any('drawdown' in t.lower() for t in titles), f"Drawdown finding not found in {titles}"

def test_portfolio_risk_agent_edge_cases(mock_analysis_context):
    from quantpits.scripts.deep_analysis.agents.portfolio_risk import PortfolioRiskAgent
    agent = PortfolioRiskAgent()
    
    # Missing metrics trigger default findings
    with patch("quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer.calculate_traditional_metrics", return_value={}), \
         patch("quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer.calculate_factor_exposure", return_value={}), \
         patch("quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer.calculate_style_exposures", return_value={}), \
         patch("quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer.calculate_holding_metrics", return_value={}), \
         patch('quantpits.scripts.analysis.utils.init_qlib'):
        
        findings = agent.analyze(mock_analysis_context)
        assert findings.agent_name == "Portfolio Risk"

# def test_model_health_agent_edge_cases(mock_analysis_context):
#     from quantpits.scripts.deep_analysis.agents.model_health import ModelHealthAgent
#     agent = ModelHealthAgent()
    
#     # 1. No performance files
#     ctx_empty = MagicMock()
#     ctx_empty.model_performance_files = []
#     findings = agent.analyze(ctx_empty)
#     assert findings.agent_name == "Model Health"
#     assert any('No model performance logs' in f.title for f in findings.findings)
    
#     # 2. Declining IC
#     with patch('json.load') as mock_json, patch('builtins.open', MagicMock()):
#         # Mock 3 snapshots with declining IC
#         mock_json.side_effect = [
#             {'all': {'IC_Mean': 0.10, 'ICIR': 1.0}},
#             {'all': {'IC_Mean': 0.05, 'ICIR': 0.5}},
#             {'all': {'IC_Mean': 0.01, 'ICIR': 0.1}}
#         ]
        
#         ctx_decl = MagicMock()
#         ctx_decl.model_performance_files = ['f1.json', 'f2.json', 'f3.json']
#         ctx_decl.window_label = 'full'
#         ctx_decl.workspace_root = mock_analysis_context.workspace_root
        
#         findings = agent.analyze(ctx_decl)
#         assert any('declining' in f.title.lower() for f in findings.findings)
