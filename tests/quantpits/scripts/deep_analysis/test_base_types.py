import pytest
from quantpits.scripts.deep_analysis.base_agent import Finding, AgentFindings, AnalysisContext
import pandas as pd

def test_finding_to_dict():
    f = Finding(
        severity="warning",
        category="Test",
        title="Test Title",
        detail="Test Detail",
        data={"metric": 1.0}
    )
    d = f.to_dict()
    assert d['severity'] == "warning"
    assert d['category'] == "Test"
    assert d['title'] == "Test Title"
    assert d['detail'] == "Test Detail"
    assert d['data']['metric'] == 1.0

def test_agent_findings_to_dict():
    f = Finding(severity="info", category="Test", title="T1", detail="D1")
    af = AgentFindings(
        agent_name="TestAgent",
        window_label="full",
        findings=[f],
        recommendations=["Rec 1"],
        raw_metrics={"m1": 100}
    )
    d = af.to_dict()
    assert d['agent_name'] == "TestAgent"
    assert d['window_label'] == "full"
    assert len(d['findings']) == 1
    assert d['findings'][0]['title'] == "T1"
    assert d['recommendations'] == ["Rec 1"]
    assert d['raw_metrics'] == {"m1": 100}

def test_analysis_context_init():
    ctx = AnalysisContext(
        start_date="2026-01-01",
        end_date="2026-01-31",
        window_label="1m",
        workspace_root="/tmp"
    )
    assert ctx.start_date == "2026-01-01"
    assert ctx.end_date == "2026-01-31"
    assert isinstance(ctx.daily_amount_df, pd.DataFrame)
    assert ctx.daily_amount_df.empty
