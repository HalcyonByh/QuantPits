import os
import json
import yaml
import pytest
from unittest.mock import patch, mock_open

# Mock the constants in train_utils before importing
@pytest.fixture(autouse=True)
def mock_env_constants(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    config_dir = workspace / "config"
    config_dir.mkdir()
    data_dir = workspace / "data"
    data_dir.mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env
    import importlib
    importlib.reload(env)

    # Now we can import train_utils
    from quantpits.scripts import train_utils
    importlib.reload(train_utils)
    
    yield train_utils, workspace

def test_load_model_registry(mock_env_constants):
    train_utils, workspace = mock_env_constants
    
    mock_registry = {
        'models': {
            'model1': {'enabled': True, 'algorithm': 'lstm'},
            'model2': {'enabled': False, 'algorithm': 'lgb'}
        }
    }
    
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_registry))):
        registry = train_utils.load_model_registry("dummy.yaml")
        assert 'model1' in registry
        assert 'model2' in registry
        assert registry['model1']['algorithm'] == 'lstm'

def test_get_enabled_models(mock_env_constants):
    train_utils, _ = mock_env_constants
    
    mock_registry = {
        'model1': {'enabled': True, 'algorithm': 'lstm'},
        'model2': {'enabled': False, 'algorithm': 'lgb'},
        'model3': {'enabled': True, 'algorithm': 'xgb'}
    }
    
    enabled = train_utils.get_enabled_models(mock_registry)
    assert len(enabled) == 2
    assert 'model1' in enabled
    assert 'model3' in enabled
    assert 'model2' not in enabled

def test_backup_file_with_date(mock_env_constants, tmp_path):
    train_utils, workspace = mock_env_constants
    
    source_file = tmp_path / "source.txt"
    source_file.write_text("hello")
    
    history_dir = tmp_path / "history"
    
    backup_path = train_utils.backup_file_with_date(
        str(source_file), 
        history_dir=str(history_dir), 
        prefix="backup_prefix"
    )
    
    assert os.path.exists(backup_path)
    assert "backup_prefix" in backup_path
    with open(backup_path, 'r') as f:
        assert f.read() == "hello"

def test_merge_train_records(mock_env_constants, tmp_path):
    train_utils, workspace = mock_env_constants
    
    record_file = tmp_path / "records.json"
    
    # Initial records
    initial = {
        "experiment_name": "exp1",
        "models": {
            "modelA": "id_A1",
            "modelB": "id_B1"
        }
    }
    record_file.write_text(json.dumps(initial))
    
    # New records to merge
    new_records = {
        "experiment_name": "exp2", # Should not override existing if we only merge models maybe? Wait, merge behavior replaces some metadata.
        "models": {
            "modelB": "id_B2", # update
            "modelC": "id_C1"  # new
        }
    }
    
    merged = train_utils.merge_train_records(new_records, record_file=str(record_file))
    
    assert merged["experiment_name"] == "exp2"
    assert merged["models"]["modelA"] == "id_A1" # Preserved
    assert merged["models"]["modelB"] == "id_B2" # Updated
    assert merged["models"]["modelC"] == "id_C1" # Added

def test_inject_config(mock_env_constants):
    train_utils, _ = mock_env_constants
    
    mock_yaml = {
        'market': 'old_market',
        'benchmark': 'old_benchmark',
        'data_handler_config': {},
        'task': {
            'dataset': {'kwargs': {'segments': {}}}
        }
    }
    
    params = {
        'freq': 'week',
        'market': 'new_market',
        'benchmark': 'new_benchmark',
        'start_time': '2000',
        'end_time': '2010',
        'fit_start_time': '2000',
        'fit_end_time': '2005',
        'valid_start_time': '2006',
        'valid_end_time': '2008',
        'test_start_time': '2009',
        'test_end_time': '2010'
    }
    
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_yaml))):
        config = train_utils.inject_config("dummy.yaml", params)
        
        assert config['market'] == 'new_market'
        assert config['benchmark'] == 'new_benchmark'
        assert config['data_handler_config']['label'] == ["Ref($close, -6) / Ref($close, -1) - 1"]
        assert config['task']['dataset']['kwargs']['segments']['train'] == ['2000', '2005']
