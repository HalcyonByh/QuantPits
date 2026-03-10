import os
import json
import yaml
from pathlib import Path

def load_workspace_config(workspace_path):
    """
    Unified configuration loader for QuantPits workspaces.
    Merges model_config.json, prod_config.json, and strategy_config.yaml.
    
    Returns a unified dict with all necessary parameters.
    """
    workspace_path = Path(workspace_path)
    config_dir = workspace_path / "config"
    
    # Files
    model_cfg_path = config_dir / "model_config.json"
    prod_cfg_path = config_dir / "prod_config.json"
    strat_cfg_path = config_dir / "strategy_config.yaml"
    
    config = {}
    
    # 1. Load Model Config (Base environment properties)
    if model_cfg_path.exists():
        with open(model_cfg_path, 'r') as f:
            config.update(json.load(f))
            
    # 2. Load Strategy Config (Single Source of Truth for strategy params)
    if strat_cfg_path.exists():
        with open(strat_cfg_path, 'r') as f:
            strat_data = yaml.safe_load(f)
            if strat_data:
                config['strategy'] = strat_data.get('strategy', {})
                config['backtest'] = strat_data.get('backtest', {})
                
                # Promote core strategy params to top-level for convenience/compatibility
                strat_params = config['strategy'].get('params', {})
                config['topk'] = strat_params.get('topk', config.get('TopK'))
                config['n_drop'] = strat_params.get('n_drop', config.get('DropN'))
                config['buy_suggestion_factor'] = strat_params.get('buy_suggestion_factor', config.get('buy_suggestion_factor'))
                
                # Compatibility mapping (Upper case versions if they don't exist)
                if 'TopK' not in config: config['TopK'] = config['topk']
                if 'DropN' not in config: config['DropN'] = config['n_drop']
                
    # 3. Load Prod Config (Current state - handles cash/holding)
    if prod_cfg_path.exists():
        with open(prod_cfg_path, 'r') as f:
            prod_data = json.load(f)
            # We only want State fields from prod_config, others should come from model/strategy
            state_fields = [
                'current_date', 'last_processed_date', 'initial_cash', 
                'current_full_cash', 'initial_holding', 'current_cash', 
                'current_holding', 'model', 'experiment_name',
                'current_train_record_id', 'current_pred_record_id'
            ]
            for field in state_fields:
                if field in prod_data:
                    config[field] = prod_data[field]
                    
    # Sanity checks / Cross-file consistency (Optional but recommended)
    # If market/benchmark exist in both, we prefer model_config but can log warnings if they mismatch
    
    return config

if __name__ == "__main__":
    # Test loading
    import sys
    if len(sys.argv) > 1:
        c = load_workspace_config(sys.argv[1])
        print(json.dumps(c, indent=2))
