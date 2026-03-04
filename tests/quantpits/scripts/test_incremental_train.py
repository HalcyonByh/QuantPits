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

# ── parse_args ───────────────────────────────────────────────────────────
import sys
import os

def test_parse_args_models(mock_env):
    inc_train = mock_env
    with patch.object(sys, 'argv', ['script.py', '--models', 'gru,mlp', '--dry-run']):
        args = inc_train.parse_args()
    assert args.models == 'gru,mlp'
    assert args.dry_run is True

def test_parse_args_list(mock_env):
    inc_train = mock_env
    with patch.object(sys, 'argv', ['script.py', '--list']):
        args = inc_train.parse_args()
    assert args.list is True

def test_parse_args_resume(mock_env):
    inc_train = mock_env
    with patch.object(sys, 'argv', ['script.py', '--all-enabled', '--resume', '--skip', 'cat']):
        args = inc_train.parse_args()
    assert args.all_enabled is True
    assert args.resume is True
    assert args.skip == 'cat'

# ── show_list ────────────────────────────────────────────────────────────
@patch('train_utils.load_model_registry')
@patch('train_utils.get_models_by_filter')
@patch('train_utils.print_model_table')
def test_show_list(mock_table, mock_filter, mock_load, mock_env):
    inc_train = mock_env
    mock_load.return_value = {
        "m1": {"enabled": True, "dataset": "Alpha158"},
        "m2": {"enabled": False, "dataset": "Alpha360"},
    }

    args = MagicMock()
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None

    inc_train.show_list(args)
    mock_table.assert_called_once()

@patch('train_utils.load_model_registry')
@patch('train_utils.get_models_by_filter')
@patch('train_utils.print_model_table')
def test_show_list_filtered(mock_table, mock_filter, mock_load, mock_env):
    inc_train = mock_env
    mock_load.return_value = {"m1": {"enabled": True, "dataset": "Alpha158"}}
    mock_filter.return_value = {"m1": {"enabled": True, "dataset": "Alpha158"}}

    args = MagicMock()
    args.algorithm = "gru"
    args.dataset = None
    args.market = None
    args.tag = None

    inc_train.show_list(args)
    mock_filter.assert_called_once()

# ── show_state ───────────────────────────────────────────────────────────
@patch('train_utils.load_run_state')
def test_show_state_none(mock_load, mock_env):
    inc_train = mock_env
    mock_load.return_value = None
    inc_train.show_state()

@patch('train_utils.load_run_state')
def test_show_state_with_data(mock_load, mock_env):
    inc_train = mock_env
    mock_load.return_value = {
        'started_at': '2020-01-01 10:00:00',
        'mode': 'incremental',
        'experiment_name': 'Prod_Train_WEEK',
        'anchor_date': '2020-01-01',
        'target_models': ['m1', 'm2', 'm3'],
        'completed': ['m1'],
        'failed': {'m2': 'ValueError'},
    }
    inc_train.show_state()

# ── run_incremental_train ────────────────────────────────────────────────
@patch('train_utils.load_model_registry')
@patch('train_utils.get_models_by_names')
@patch('train_utils.get_enabled_models')
@patch('train_utils.get_models_by_filter')
@patch('train_utils.print_model_table')
def test_run_incremental_train_no_targets(mock_table, mock_filter, mock_enabled, mock_names, mock_load, mock_env):
    inc_train = mock_env
    args = MagicMock()
    args.models = None
    args.all_enabled = False
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = None

    inc_train.run_incremental_train(args)
    # Should print error and return

@patch('train_utils.load_model_registry')
@patch('train_utils.get_models_by_names')
@patch('train_utils.get_enabled_models')
@patch('train_utils.get_models_by_filter')
@patch('train_utils.print_model_table')
def test_run_incremental_train_dry_run(mock_table, mock_filter, mock_enabled, mock_names, mock_load, mock_env):
    inc_train = mock_env
    mock_load.return_value = {"m1": {"yaml_file": "m1.yaml"}}
    mock_names.return_value = {"m1": {"yaml_file": "m1.yaml"}}

    args = MagicMock()
    args.models = "m1"
    args.all_enabled = False
    args.algorithm = None
    args.dataset = None
    args.market = None
    args.tag = None
    args.skip = None
    args.resume = False
    args.dry_run = True

    inc_train.run_incremental_train(args)
    # Should print dry-run message and return without training

# ── main ─────────────────────────────────────────────────────────────────
def test_main_no_selection(mock_env):
    inc_train = mock_env
    with patch.object(sys, 'argv', ['script.py']):
        inc_train.main()

def test_main_list(mock_env):
    inc_train = mock_env
    with patch.object(sys, 'argv', ['script.py', '--list']):
        with patch.object(inc_train, 'show_list') as mock_show:
            inc_train.main()
            mock_show.assert_called_once()

def test_main_show_state(mock_env):
    inc_train = mock_env
    with patch.object(sys, 'argv', ['script.py', '--show-state']):
        with patch.object(inc_train, 'show_state') as mock_show:
            inc_train.main()
            mock_show.assert_called_once()

@patch('train_utils.clear_run_state')
def test_main_clear_state(mock_clear, mock_env):
    inc_train = mock_env
    with patch.object(sys, 'argv', ['script.py', '--clear-state']):
        inc_train.main()
        mock_clear.assert_called_once()

