import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, incremental_train
    import importlib
    importlib.reload(env)
    importlib.reload(incremental_train)
    
    yield incremental_train

@patch('train_utils.get_models_by_names')
@patch('train_utils.get_enabled_models')
@patch('train_utils.get_models_by_filter')
@patch('train_utils.load_model_registry')
def test_resolve_target_models_by_names(mock_load, mock_filter, mock_enabled, mock_names, mock_env):
    inc_train = mock_env
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
    
    targets = inc_train.resolve_target_models(args)
    assert targets == {"m1": {}}
    mock_names.assert_called_once_with(["m1"], {"m1": {}, "m2": {}})

@patch('train_utils.get_models_by_names')
@patch('train_utils.get_enabled_models')
@patch('train_utils.get_models_by_filter')
@patch('train_utils.load_model_registry')
def test_resolve_target_models_all_enabled(mock_load, mock_filter, mock_enabled, mock_names, mock_env):
    inc_train = mock_env
    mock_load.return_value = {"m1": {}, "m2": {}}
    mock_enabled.return_value = {"m1": {}, "m2": {}}
    
    args = MagicMock()
    args.models = None
    args.all_enabled = True
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = None
    
    targets = inc_train.resolve_target_models(args)
    assert len(targets) == 2
    mock_enabled.assert_called_once()

@patch('train_utils.get_models_by_names')
@patch('train_utils.get_enabled_models')
@patch('train_utils.get_models_by_filter')
@patch('train_utils.load_model_registry')
def test_resolve_target_models_by_filter(mock_load, mock_filter, mock_enabled, mock_names, mock_env):
    inc_train = mock_env
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
    
    targets = inc_train.resolve_target_models(args)
    assert targets == {"m1": {}}
    mock_filter.assert_called_once_with({"m1": {}, "m2": {}}, algorithm="lstm", dataset=None, market=None, tag=None)

@patch('train_utils.get_models_by_names')
@patch('train_utils.get_enabled_models')
@patch('train_utils.get_models_by_filter')
@patch('train_utils.load_model_registry')
def test_resolve_target_models_skip(mock_load, mock_filter, mock_enabled, mock_names, mock_env):
    inc_train = mock_env
    mock_load.return_value = {"m1": {}, "m2": {}}
    mock_enabled.return_value = {"m1": {}, "m2": {}}
    
    args = MagicMock()
    args.models = None
    args.all_enabled = True
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = "m2"
    
    targets = inc_train.resolve_target_models(args)
    assert targets == {"m1": {}}

@patch('train_utils.load_model_registry')
def test_resolve_target_models_none(mock_load, mock_env):
    inc_train = mock_env
    mock_load.return_value = {"m1": {}, "m2": {}}
    args = MagicMock()
    args.models = None
    args.all_enabled = False
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = None
    
    targets = inc_train.resolve_target_models(args)
    assert targets is None
