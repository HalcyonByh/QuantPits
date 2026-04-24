import pytest
from quantpits.scripts.deep_analysis.synthesizer import Synthesizer
from quantpits.scripts.deep_analysis.base_agent import AgentFindings, Finding

def test_synthesizer_cross_reference_alpha_decay():
    # Case: Model IC declining + Market regime high-vol
    f_model = Finding(severity='warning', category='Model Health', 
                      title='Model IC declining', detail='...')
    af_model = AgentFindings(agent_name='Model Health', window_label='1m', 
                             findings=[f_model])
    
    af_market = AgentFindings(agent_name='Market Regime', window_label='1m', 
                              raw_metrics={'regime': 'High-Vol'})
    
    synth = Synthesizer([af_model, af_market])
    result = synth.synthesize()
    
    # Should find the cross-finding
    cross = [f for f in result['cross_findings'] if 'IC decline' in f.title]
    assert len(cross) == 1
    assert cross[0].severity == 'warning'

def test_synthesizer_health_status():
    # Healthy case
    synth_ok = Synthesizer([])
    assert "HEALTHY" in synth_ok.synthesize()['health_status']
    
    # Critical case
    f_crit = Finding(severity='critical', category='Test', title='T', detail='D')
    af_crit = AgentFindings(agent_name='A1', window_label='1m', findings=[f_crit, f_crit])
    synth_crit = Synthesizer([af_crit])
    assert "CRITICAL" in synth_crit.synthesize()['health_status']

def test_synthesizer_recommendations():
    f = Finding(severity='critical', category='Cross-Agent', title='T', detail='D1')
    af = AgentFindings(agent_name='A1', window_label='1m', recommendations=['R1'])
    
    # Simulate a cross-finding by manually adding it or triggering a rule
    # Here we just check if it collects recommendations from agents
    synth = Synthesizer([af])
    result = synth.synthesize()
    
    assert any(r['text'] == 'R1' for r in result['recommendations'])
