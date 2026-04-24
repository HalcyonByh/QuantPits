import pytest
import os
import pandas as pd
from quantpits.scripts.deep_analysis.coordinator import Coordinator, _extract_date

def test_extract_date():
    assert _extract_date("model_performance_2026-04-20.json") == "2026-04-20"
    assert _extract_date("2026-04-20-some_file.csv") == "2026-04-20"
    assert _extract_date("buy_suggestion_ensemble_2026-04-20_123456.csv") == "2026-04-20"
    assert _extract_date("random_file.txt") is None

def test_coordinator_discovery(mock_workspace):
    coord = Coordinator(mock_workspace)
    coord.discover()
    
    assert coord._data_start_date == "2026-01-01"
    assert coord._data_end_date == "2026-04-10"  # 2026-01-01 + 99 days
    assert not coord._daily_amount_df.empty

def test_coordinator_generate_windows(mock_workspace):
    coord = Coordinator(mock_workspace, freq_change_date="2026-02-01")
    coord.discover()
    windows = coord.generate_windows()
    
    labels = [w['label'] for w in windows]
    assert 'full' in labels
    assert 'weekly_era' in labels
    assert '1m' in labels
    
    full_window = next(w for w in windows if w['label'] == 'full')
    assert full_window['start_date'] == "2026-01-01"
    
    weekly_window = next(w for w in windows if w['label'] == 'weekly_era')
    assert weekly_window['start_date'] == "2026-02-01"

def test_coordinator_build_context(mock_workspace):
    coord = Coordinator(mock_workspace)
    coord.discover()
    
    window = {
        'label': 'test',
        'start_date': '2026-02-01',
        'end_date': '2026-02-28',
        'is_pre_cutoff': False
    }
    ctx = coord.build_context(window)
    
    assert ctx.start_date == '2026-02-01'
    assert ctx.window_label == 'test'
    # Check slicing: 2026-02-01 to 2026-02-28 is 28 days
    assert len(ctx.daily_amount_df) == 28
