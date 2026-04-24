import sys
from unittest.mock import MagicMock, patch

# Mock openai module before importing LLMInterface
mock_openai = MagicMock()
sys.modules["openai"] = mock_openai

from quantpits.scripts.deep_analysis.llm_interface import LLMInterface

def test_llm_interface_template_fallback():
    # Test fallback when no API key
    interface = LLMInterface(api_key=None)
    interface._client = None 
    
    with patch.dict('os.environ', {}, clear=True):
        interface.api_key = None
        assert interface.is_available() is False
    
    # Test with missing data to hit branches in _template_summary
    synthesis_result = {
        'health_status': 'Healthy',
        'executive_summary_data': {
            'windows_analyzed': ['2026-01-01'],
            'agents_run': ['AgentA'],
            'critical_count': 0,
            'warning_count': 1,
            'positive_count': 2,
            # 'market_regime': None,  # Missing
            # 'cagr_1y': None,        # Missing
        },
        'recommendations': [
            {'priority': 'P0', 'text': 'Do something', 'source': 'AgentA'}
        ],
        'cross_findings': [
             MagicMock(severity='critical', title='Major Issue'),
             MagicMock(severity='warning', title='Minor Issue'),
             MagicMock(severity='info', title='Info Issue')
        ]
    }
    
    summary = interface.generate_executive_summary(synthesis_result)
    assert "**System Health:** Healthy" in summary
    assert "Do something" in summary
    assert "🔴 Major Issue" in summary
    assert "🟡 Minor Issue" in summary
    assert "🟢 Info Issue" in summary

def test_llm_interface_openai():
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "LLM Generated Summary"
    mock_client.chat.completions.create.return_value = mock_response
    
    interface = LLMInterface(api_key="fake_key", base_url="http://fake.url")
    interface._get_client()
    
    assert interface.is_available() is True
    
    synthesis_result = {
        'health_status': 'Healthy',
        'executive_summary_data': {
            'windows_analyzed': ['2026-01-01'],
            'agents_run': ['AgentA'],
            'critical_count': 0,
            'warning_count': 0,
            'positive_count': 0,
            'market_regime': 'Bull',
            'cagr_1y': 0.1,
            'sharpe_1y': 1.0
        },
        'cross_findings': [
            MagicMock(severity='critical', title='Major Issue', detail='Detail here')
        ],
        'recommendations': [
            {'priority': 'P0', 'text': 'Rec 1', 'source': 'AgentA'}
        ],
        'change_impact': [
            {'event': {'type': 'retrain', 'date': '2026-01-15', 'model': 'M1'}}
        ],
        'external_notes': 'Some notes'
    }
    
    summary = interface.generate_executive_summary(synthesis_result)
    assert summary == "LLM Generated Summary"
    mock_client.chat.completions.create.assert_called_once()

def test_llm_interface_error_fallback():
    interface = LLMInterface(api_key="fake_key")
    mock_client = MagicMock()
    interface._client = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    
    synthesis_result = {'health_status': 'Error State'}
    
    summary = interface.generate_executive_summary(synthesis_result)
    assert "**System Health:** Error State" in summary

def test_llm_interface_import_error():
    # Test ImportError for openai
    interface = LLMInterface(api_key="fake_key")
    interface._client = None
    
    with patch.dict(sys.modules, {'openai': None}):
        import pytest
        with pytest.raises(RuntimeError, match="openai package not installed"):
            interface._get_client()
