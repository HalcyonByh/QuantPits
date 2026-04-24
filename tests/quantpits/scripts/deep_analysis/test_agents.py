import pytest
import pandas as pd
from unittest.mock import patch
from quantpits.scripts.deep_analysis.agents.market_regime import MarketRegimeAgent
from quantpits.scripts.deep_analysis.agents.model_health import ModelHealthAgent
from quantpits.scripts.deep_analysis.agents.portfolio_risk import PortfolioRiskAgent
from quantpits.scripts.deep_analysis.agents.execution_quality import ExecutionQualityAgent
from quantpits.scripts.deep_analysis.agents.ensemble_eval import EnsembleEvolutionAgent
from quantpits.scripts.deep_analysis.agents.prediction_audit import PredictionAuditAgent
from quantpits.scripts.deep_analysis.agents.trade_pattern import TradePatternAgent

def test_market_regime_agent(mock_analysis_context):
    agent = MarketRegimeAgent()
    findings = agent.analyze(mock_analysis_context)
    
    assert findings.agent_name == "Market Regime"
    assert 'regime' in findings.raw_metrics
    assert len(findings.findings) > 0

def test_model_health_agent(mock_analysis_context, mock_env_constants):
    agent = ModelHealthAgent()
    findings = agent.analyze(mock_analysis_context)
    
    assert findings.agent_name == "Model Health"
    assert isinstance(findings.findings, list)
    assert 'scorecard' in findings.raw_metrics

@patch("quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer.calculate_traditional_metrics")
@patch("quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer.calculate_factor_exposure")
@patch("quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer.calculate_style_exposures")
@patch("quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer.calculate_holding_metrics")
def test_portfolio_risk_agent(mock_hold, mock_style, mock_factor, mock_trad, mock_analysis_context, mock_env_constants):
    mock_trad.return_value = {
        'CAGR_252': 0.15, 'Sharpe': 1.5, 'Max_Drawdown': -0.05,
        'Calmar': 3.0, 'Excess_Return_CAGR_252': 0.05, 'Sortino': 2.0
    }
    mock_factor.return_value = {
        'Beta_Market': 1.1, 'Annualized_Alpha': 0.05,
        'Annualized_Alpha_t': 2.5, 'Annualized_Alpha_p': 0.01,
        'Beta_Market_t': 10.0, 'Beta_Market_p': 0.001, 'R_Squared': 0.8
    }
    mock_style.return_value = {
        'Multi_Factor_Intercept': 0.03, 'Multi_Factor_Intercept_t': 2.0, 'Multi_Factor_Intercept_p': 0.04,
        'Barra_Liquidity_Exp_(High-Low)': 0.1, 'Barra_Momentum_Exp_(High-Low)': 0.2, 'Barra_Volatility_Exp_(High-Low)': -0.1
    }
    mock_hold.return_value = {
        'Avg_Daily_Holdings_Count': 20, 'Avg_Top1_Concentration': 0.05, 'Daily_Holding_Win_Rate': 0.55
    }

    agent = PortfolioRiskAgent()
    findings = agent.analyze(mock_analysis_context)
    
    assert findings.agent_name == "Portfolio Risk"
    assert isinstance(findings.findings, list)
    assert 'traditional' in findings.raw_metrics
    assert findings.raw_metrics['traditional']['Sharpe'] == 1.5

@patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.calculate_slippage_and_delay")
@patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_explicit_costs")
@patch("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_order_discrepancies")
def test_execution_quality_agent(mock_discrepancies, mock_costs, mock_slippage, mock_analysis_context, mock_env_constants):
    mock_slippage.return_value = pd.DataFrame({
        '交易类别': ['上海A股普通股票竞价买入', '上海A股普通股票竞价卖出'],
        '成交金额': [10000, 10000],
        'Delay_Cost': [0.001, 0.002],
        'Exec_Slippage': [0.003, 0.001],
        'ADV_Participation_Rate': [0.02, 0.01]
    })
    mock_costs.return_value = {
        'fee_ratio': 0.0005,
        'total_fees': 10.0,
        'total_dividend': 5.0
    }
    mock_discrepancies.return_value = {
        'theoretical_substitute_bias_impact': -0.01,
        'realized_substitute_bias_impact': -0.03,
        'total_missed_count': 5,
        'total_substitute_count': 5
    }

    agent = ExecutionQualityAgent()
    findings = agent.analyze(mock_analysis_context)
    
    assert findings.agent_name == "Execution Quality"
    assert isinstance(findings.findings, list)
    assert 'buy_total_friction' in findings.raw_metrics
    # 0.001 delay + 0.003 slippage = 0.004
    assert abs(findings.raw_metrics['buy_total_friction'] - 0.004) < 1e-6

def test_ensemble_eval_agent(mock_analysis_context):
    agent = EnsembleEvolutionAgent()
    findings = agent.analyze(mock_analysis_context)
    
    assert findings.agent_name == "Ensemble Evolution"
    assert 'combo_trends' in findings.raw_metrics
    assert len(findings.findings) > 0

@patch("qlib.data.D.features")
def test_prediction_audit_agent(mock_features, mock_analysis_context):
    # Mock D.features to return a MultiIndex DataFrame like Qlib
    mock_features.return_value = pd.DataFrame(
        {'Ref($close, -5) / $close - 1': [0.05, -0.02]},
        index=pd.MultiIndex.from_product([['2026-03-20'], ['600000.SH', '000001.SZ']], names=['datetime', 'instrument'])
    )
    agent = PredictionAuditAgent()
    findings = agent.analyze(mock_analysis_context)
    
    assert findings.agent_name == "Prediction Audit"
    assert 'buy_hit_rate' in findings.raw_metrics
    assert 'consensus_analysis' in findings.raw_metrics

def test_trade_pattern_agent(mock_analysis_context):
    agent = TradePatternAgent()
    findings = agent.analyze(mock_analysis_context)
    
    assert findings.agent_name == "Trade Pattern"
    assert 'trade_counts' in findings.raw_metrics
