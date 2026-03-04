import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, prod_predict_only
    import importlib
    importlib.reload(env)
    importlib.reload(prod_predict_only)
    
    yield prod_predict_only

@patch('train_utils.get_models_by_names')
@patch('train_utils.get_enabled_models')
@patch('train_utils.get_models_by_filter')
@patch('train_utils.load_model_registry')
def test_resolve_target_models_by_names(mock_load, mock_filter, mock_enabled, mock_names, mock_env):
    ppo = mock_env
    mock_load.return_value = {"m1": {}, "m2": {}}
    mock_names.return_value = {"m1": {}}
    
    args = MagicMock()
    args.models = "m1"
    args.all_enabled = False
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = None
    
    targets = ppo.resolve_target_models(args)
    assert targets == {"m1": {}}

@patch('train_utils.get_models_by_names')
@patch('train_utils.get_enabled_models')
@patch('train_utils.get_models_by_filter')
@patch('train_utils.load_model_registry')
def test_resolve_target_models_by_filter(mock_load, mock_filter, mock_enabled, mock_names, mock_env):
    ppo = mock_env
    mock_load.return_value = {"m1": {}, "m2": {}}
    mock_filter.return_value = {"m1": {}}
    
    args = MagicMock()
    args.models = None
    args.all_enabled = False
    args.algorithm = "lstm"
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = None
    
    targets = ppo.resolve_target_models(args)
    assert targets == {"m1": {}}

@patch('train_utils.load_model_registry')
def test_resolve_target_models_none(mock_load, mock_env):
    ppo = mock_env
    mock_load.return_value = {"m1": {}, "m2": {}}
    args = MagicMock()
    args.models = None
    args.all_enabled = False
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = None
    
    targets = ppo.resolve_target_models(args)
    assert targets is None
