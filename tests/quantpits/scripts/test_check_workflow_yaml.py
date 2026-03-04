import os
import yaml
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    config_dir = workspace / "config"
    config_dir.mkdir()
    
    import sys
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.scripts import env, check_workflow_yaml
    import importlib
    importlib.reload(env)
    importlib.reload(check_workflow_yaml)
    
    monkeypatch.setattr(check_workflow_yaml, 'ROOT_DIR', str(workspace))
    monkeypatch.setattr(check_workflow_yaml, 'CONFIG_DIR', str(config_dir))
    
    yield check_workflow_yaml, workspace

# ── check_yamls ──────────────────────────────────────────────────────────
def test_check_yamls_valid_week(mock_env):
    cw, workspace = mock_env
    valid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -6) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 52
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "week"
"""
    (workspace / "config" / "workflow_config_m1.yaml").write_text(valid_yaml)
    anomalies = cw.check_yamls(freq="week")
    assert anomalies == {}

def test_check_yamls_wrong_label(mock_env):
    cw, workspace = mock_env
    invalid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 52
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "week"
"""
    (workspace / "config" / "workflow_config_m1.yaml").write_text(invalid_yaml)
    anomalies = cw.check_yamls(freq="week")
    assert "workflow_config_m1.yaml" in anomalies
    assert any("LABEL:" in val for val in anomalies["workflow_config_m1.yaml"])

def test_check_yamls_wrong_time_per_step(mock_env):
    cw, workspace = mock_env
    invalid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -6) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 52
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "day"
"""
    (workspace / "config" / "workflow_config_m1.yaml").write_text(invalid_yaml)
    anomalies = cw.check_yamls(freq="week")
    assert any("TIME_PER_STEP:" in val for val in anomalies["workflow_config_m1.yaml"])

def test_check_yamls_wrong_ann_scaler(mock_env):
    cw, workspace = mock_env
    invalid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -6) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 252
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "week"
"""
    (workspace / "config" / "workflow_config_m1.yaml").write_text(invalid_yaml)
    anomalies = cw.check_yamls(freq="week")
    assert any("ANN_SCALER:" in val for val in anomalies["workflow_config_m1.yaml"])

def test_check_yamls_day_freq(mock_env):
    cw, workspace = mock_env
    valid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 252
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "day"
"""
    (workspace / "config" / "workflow_config_m1.yaml").write_text(valid_yaml)
    anomalies = cw.check_yamls(freq="day")
    assert anomalies == {}

# ── fix_yamls ────────────────────────────────────────────────────────────
def test_fix_yamls_label(mock_env):
    cw, workspace = mock_env
    target_yaml = workspace / "config" / "workflow_config_m1.yaml"
    invalid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
"""
    target_yaml.write_text(invalid_yaml)
    cw.fix_yamls(freq="week")
    
    fixed_yaml = target_yaml.read_text()
    assert 'label: ["Ref($close, -6) / Ref($close, -1) - 1"]' in fixed_yaml

def test_fix_yamls_ann_scaler(mock_env):
    cw, workspace = mock_env
    target_yaml = workspace / "config" / "workflow_config_m1.yaml"
    invalid_yaml = """
task:
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 252
"""
    target_yaml.write_text(invalid_yaml)
    cw.fix_yamls(freq="week")
    
    fixed_yaml = target_yaml.read_text()
    assert 'ann_scaler: 52' in fixed_yaml

def test_fix_yamls_lr_scientific(mock_env):
    cw, workspace = mock_env
    target_yaml = workspace / "config" / "workflow_config_m1.yaml"
    invalid_yaml = """
task:
  model:
    kwargs:
      lr: 1e-4
"""
    target_yaml.write_text(invalid_yaml)
    
    # Needs to report it first
    anomalies = cw.check_yamls(freq="week")
    assert any("LR" in val for val in anomalies.get("workflow_config_m1.yaml", []))
    
    cw.fix_yamls(freq="week")
    
    fixed_yaml = target_yaml.read_text()
    assert 'lr: 0.0001' in fixed_yaml
