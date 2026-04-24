import pytest
import pandas as pd
import numpy as np
import os
import json
from unittest.mock import patch, MagicMock
from quantpits.scripts.deep_analysis.agents.prediction_audit import PredictionAuditAgent
from quantpits.scripts.deep_analysis.base_agent import AnalysisContext

@pytest.fixture
def agent():
    return PredictionAuditAgent()

def test_analyze_hit_rate_thresholds(agent, mock_analysis_context):
    """Test analyze method with different hit rates to cover lines 33-56 and 67-71."""
    
    # 1. Low buy hit rate (< 40%)
    with patch.object(agent, '_analyze_suggestion_hits') as mock_hits, \
         patch.object(agent, '_analyze_consensus', return_value={}), \
         patch.object(agent, '_analyze_holding_retrospective', return_value={}):
        
        # direction='buy'
        mock_hits.side_effect = lambda ctx, direction: {
            'overall': {
                'hit_rate': 0.3,
                'n_suggestions': 10,
                'avg_return': -0.01
            }
        } if direction == 'buy' else {
            'overall': {
                'hit_rate': 0.5,
                'n_suggestions': 10
            }
        }
        
        findings = agent.analyze(mock_analysis_context)
        titles = [f.title for f in findings.findings]
        assert 'Low buy suggestion hit rate' in titles
        assert any("Buy suggestion hit rate is below 40%" in r for r in findings.recommendations)

    # 2. High buy hit rate (> 60%)
    with patch.object(agent, '_analyze_suggestion_hits') as mock_hits, \
         patch.object(agent, '_analyze_consensus', return_value={}), \
         patch.object(agent, '_analyze_holding_retrospective', return_value={}):
        
        mock_hits.side_effect = lambda ctx, direction: {
            'overall': {
                'hit_rate': 0.7,
                'n_suggestions': 10,
                'avg_return': 0.02
            }
        } if direction == 'buy' else None
        
        findings = agent.analyze(mock_analysis_context)
        titles = [f.title for f in findings.findings]
        assert 'Strong buy suggestion accuracy' in titles

def test_analyze_consensus_logic(agent, mock_analysis_context):
    """Test analyze method consensus findings to cover lines 89-104."""
    
    # Divergence outperforming consensus
    with patch.object(agent, '_analyze_suggestion_hits', return_value={}), \
         patch.object(agent, '_analyze_consensus') as mock_cons, \
         patch.object(agent, '_analyze_holding_retrospective', return_value={}):
        
        mock_cons.return_value = {
            'high_consensus_avg_return': 0.01,
            'high_divergence_avg_return': 0.05,
            'n_high_consensus': 5,
            'n_high_divergence': 5
        }
        
        findings = agent.analyze(mock_analysis_context)
        titles = [f.title for f in findings.findings]
        assert 'Divergence picks outperform consensus' in titles

def test_analyze_suggestion_hits_logic(agent, mock_analysis_context, tmp_path):
    """Test _analyze_suggestion_hits internal logic to cover lines 187-188, 191, 197-199."""
    
    with patch('quantpits.scripts.analysis.utils.init_qlib'), \
         patch('qlib.data.D') as mock_d:
        
        # 1. Test future_dates logic (line 187-188)
        # We need fwd_returns to have a date > date_str
        date_str = '2026-03-20'
        future_date = pd.Timestamp('2026-03-25')
        mock_d.features.return_value = pd.DataFrame(
            {'fwd_return': [0.05]},
            index=pd.MultiIndex.from_product([[future_date], ['600000.SH']], names=['datetime', 'instrument'])
        )
        
        ctx = MagicMock(spec=AnalysisContext)
        ctx.buy_suggestion_files = [str(mock_analysis_context.buy_suggestion_files[0])] # 2026-03-20
        
        res = agent._analyze_suggestion_hits(ctx, direction='buy')
        assert 'overall' in res
        assert res['overall']['hit_rate'] == 1.0

        # 2. Test fwd_on_date.empty (line 191)
        # Return features with a date BEFORE the requested date
        mock_d.features.return_value = pd.DataFrame(
            {'fwd_return': [0.05]},
            index=pd.MultiIndex.from_product([[pd.Timestamp('2026-03-10')], ['600000.SH']], names=['datetime', 'instrument'])
        )
        res = agent._analyze_suggestion_hits(ctx, direction='buy')
        assert res == {}

def test_analyze_suggestion_hits_edge_cases(agent, mock_analysis_context, tmp_path):
    """Test _analyze_suggestion_hits edge cases to cover lines 138, 144-145, 154, 159-161, 165."""
    
    # 1. No files (line 138)
    ctx_no_files = MagicMock(spec=AnalysisContext)
    ctx_no_files.buy_suggestion_files = []
    res = agent._analyze_suggestion_hits(ctx_no_files, direction='buy')
    assert res == {}

    # 2. Qlib init failure (lines 144-145)
    with patch('quantpits.scripts.analysis.utils.init_qlib', side_effect=Exception("Failed")):
        ctx = MagicMock(spec=AnalysisContext)
        ctx.buy_suggestion_files = ['/path/to/buy_suggestion_2026-03-20.csv']
        res = agent._analyze_suggestion_hits(ctx)
        assert res == {'error': 'Qlib initialization failed'}

    # 3. Filename no date (line 154)
    with patch('quantpits.scripts.analysis.utils.init_qlib'), patch('qlib.data.D'):
        ctx = MagicMock(spec=AnalysisContext)
        ctx.buy_suggestion_files = ['invalid_name.csv']
        res = agent._analyze_suggestion_hits(ctx)
        assert res == {}

    # 4. CSV missing 'instrument' or exception (lines 159-161, 165)
    with patch('quantpits.scripts.analysis.utils.init_qlib'), patch('qlib.data.D'):
        ctx = MagicMock(spec=AnalysisContext)
        # Using a name with date for bad_csv
        bad_csv_named = tmp_path / "buy_suggestion_2026-03-20.csv"
        bad_csv_named.write_text("col1,col2\n1,2")
        
        empty_csv_named = tmp_path / "buy_suggestion_2026-03-21.csv"
        empty_csv_named.write_text("instrument\n")
        
        ctx.buy_suggestion_files = [str(bad_csv_named), str(empty_csv_named)]
        res = agent._analyze_suggestion_hits(ctx)
        assert res == {}

def test_analyze_consensus_more_edge_cases(agent, tmp_path):
    """Test more edge cases in _analyze_consensus to cover lines 274-275, 294."""
    with patch('quantpits.scripts.analysis.utils.init_qlib'), \
         patch('qlib.data.D'):
        
        json_file = tmp_path / "model_opinions_2026-03-20.json"
        json_file.write_text("{}")
        
        # 1. Exception during pd.read_csv (lines 274-275)
        csv_file = tmp_path / "model_opinions_2026-03-20.csv"
        csv_file.write_text("invalid")
        
        ctx = MagicMock(spec=AnalysisContext)
        ctx.model_opinions_files = [str(json_file)]
        
        with patch('quantpits.scripts.deep_analysis.agents.prediction_audit.pd.read_csv', side_effect=Exception("Read error")):
             res = agent._analyze_consensus(ctx)
             assert res == {'n_high_consensus': 0, 'n_high_divergence': 0}

        # 2. total_models == 0 (line 294)
        # We need a valid DF but no model columns
        df_no_models = pd.DataFrame({'instrument': ['600000.SH']})
        with patch('quantpits.scripts.deep_analysis.agents.prediction_audit.pd.read_csv', return_value=df_no_models):
             res = agent._analyze_consensus(ctx)
             assert res == {'n_high_consensus': 0, 'n_high_divergence': 0}

def test_analyze_suggestion_hits_read_csv_exception(agent, tmp_path):
    """Test exception in pd.read_csv for _analyze_suggestion_hits (lines 160-161)."""
    with patch('quantpits.scripts.analysis.utils.init_qlib'), \
         patch('qlib.data.D'):
        
        ctx = MagicMock(spec=AnalysisContext)
        ctx.buy_suggestion_files = ['buy_suggestion_2026-03-20.csv']
        
        with patch('quantpits.scripts.deep_analysis.agents.prediction_audit.pd.read_csv', side_effect=Exception("Read error")):
             res = agent._analyze_suggestion_hits(ctx)
             assert res == {}

def test_analyze_positive_consensus(agent, mock_analysis_context):
    """Test positive consensus finding (line 98)."""
    with patch.object(agent, '_analyze_suggestion_hits', return_value={}), \
         patch.object(agent, '_analyze_consensus') as mock_cons, \
         patch.object(agent, '_analyze_holding_retrospective', return_value={}):
        
        mock_cons.return_value = {
            'high_consensus_avg_return': 0.10,
            'high_divergence_avg_return': 0.01,
            'n_high_consensus': 5,
            'n_high_divergence': 5
        }
        
        findings = agent.analyze(mock_analysis_context)
        titles = [f.title for f in findings.findings]
        assert 'Consensus picks outperform' in titles

def test_analyze_consensus_edge_cases(agent, mock_analysis_context, tmp_path):
    """Test _analyze_consensus edge cases to cover lines 241, 247-248, 256, 261-262, 268, 273-275, 281, 294, 307."""
    
    # 1. No files (line 241)
    ctx = MagicMock(spec=AnalysisContext)
    ctx.model_opinions_files = []
    assert agent._analyze_consensus(ctx) == {}

    # 2. Qlib init failure (lines 247-248)
    with patch('quantpits.scripts.analysis.utils.init_qlib', side_effect=Exception()):
        ctx.model_opinions_files = ['f.json']
        assert agent._analyze_consensus(ctx) == {'error': 'Qlib init failed'}

    # 3. Filename no date (line 256)
    with patch('quantpits.scripts.analysis.utils.init_qlib'):
        ctx.model_opinions_files = ['invalid.json']
        assert agent._analyze_consensus(ctx) == {'n_high_consensus': 0, 'n_high_divergence': 0}

    # 4. JSON load error (lines 261-262)
    with patch('quantpits.scripts.analysis.utils.init_qlib'):
        bad_json = tmp_path / "model_opinions_2026-03-20.json"
        bad_json.write_text("invalid json")
        ctx.model_opinions_files = [str(bad_json)]
        assert agent._analyze_consensus(ctx) == {'n_high_consensus': 0, 'n_high_divergence': 0}

    # 5. Missing CSV (line 268)
    with patch('quantpits.scripts.analysis.utils.init_qlib'):
        good_json = tmp_path / "model_opinions_2026-03-21.json"
        good_json.write_text("{}")
        # CSV is missing
        ctx.model_opinions_files = [str(good_json)]
        assert agent._analyze_consensus(ctx) == {'n_high_consensus': 0, 'n_high_divergence': 0}

    # 6. CSV missing instrument or no model cols or total models 0 (lines 273-275, 281, 294)
    with patch('quantpits.scripts.analysis.utils.init_qlib'):
        json_file = tmp_path / "model_opinions_2026-03-22.json"
        json_file.write_text("{}")
        
        # Missing instrument
        csv_file = tmp_path / "model_opinions_2026-03-22.csv"
        csv_file.write_text("col1,col2\n1,2")
        ctx.model_opinions_files = [str(json_file)]
        assert agent._analyze_consensus(ctx) == {'n_high_consensus': 0, 'n_high_divergence': 0}
        
        # No model_ cols
        csv_file.write_text("instrument\n600000.SH")
        assert agent._analyze_consensus(ctx) == {'n_high_consensus': 0, 'n_high_divergence': 0}

    # 7. No high cons/div picks (line 307)
    with patch('quantpits.scripts.analysis.utils.init_qlib'):
        json_file = tmp_path / "model_opinions_2026-03-23.json"
        json_file.write_text("{}")
        csv_file = tmp_path / "model_opinions_2026-03-23.csv"
        # middle ground consensus ratio (0.5) - will go into divergence if > 0.3 and < 0.7
        # To avoid both, we can make it exactly 0.8 or 0.2?
        # consensus_ratio > 0.8 is high_cons
        # 0.3 < ratio < 0.7 is high_div
        # So ratio = 0.75 should avoid both
        csv_file.write_text("instrument,model_1,model_2,model_3,model_4\n600000.SH,BUY,BUY,BUY,SELL") # 3/4 = 0.75
        ctx.model_opinions_files = [str(json_file)]
        assert agent._analyze_consensus(ctx) == {'n_high_consensus': 0, 'n_high_divergence': 0}

def test_analyze_holding_retrospective_edge_cases(agent):
    """Test _analyze_holding_retrospective edge cases to cover lines 352, 356, 361, 368, 380."""
    
    # 1. Empty holding_log_df (line 352)
    ctx = MagicMock(spec=AnalysisContext)
    ctx.holding_log_df = pd.DataFrame()
    assert agent._analyze_holding_retrospective(ctx) == {}

    # 2. Missing columns (line 356)
    ctx.holding_log_df = pd.DataFrame({'col1': [1]})
    assert agent._analyze_holding_retrospective(ctx) == {}

    # 3. DF empty after CASH filter (line 361)
    ctx.holding_log_df = pd.DataFrame({
        '证券代码': ['CASH'],
        '成交日期': [pd.Timestamp('2026-03-20')]
    })
    assert agent._analyze_holding_retrospective(ctx) == {}

    # 4. Missing 浮盈收益率 (line 380)
    ctx.holding_log_df = pd.DataFrame({
        '证券代码': ['600000.SH'],
        '成交日期': [pd.Timestamp('2026-03-20')]
    })
    res = agent._analyze_holding_retrospective(ctx)
    assert 'summary' in res
    assert 'holdings as of 2026-03-20' in res['summary']
    assert 'Win rate' not in res['summary']

    # 5. Empty current_holdings after date filter (line 368) - Hard to trigger since we take max date
    # But if latest_date is NaT? 
    # Actually if latest_date = df['成交日期'].max() and df is not empty, current_holdings shouldn't be empty.
    # Unless... wait. If latest_date is somehow not in the dates? impossible.
    # What if we mock the max() to return something not in the df?
    with patch('pandas.Series.max', return_value=pd.Timestamp('2099-01-01')):
         assert agent._analyze_holding_retrospective(ctx) == {}
