import pytest
import os
import sys

@pytest.fixture(autouse=True)
def prevent_mlruns(monkeypatch, tmp_path):
    """
    Globally redirect MLflowExpManager's URI to a temporary directory so that
    unmocked instances of Qlib workflow recorder won't create 'mlruns' in the project root.
    This is safer than mocking the entire class because it preserves object types.
    """
    import qlib.workflow.expm
    original_init = qlib.workflow.expm.MLflowExpManager.__init__
    
    def mocked_init(self, uri, default_exp_name, *args, **kwargs):
        # Force the URI to be in the tmp_path to avoid creating mlruns in root
        safe_uri = f"file://{tmp_path / 'mock_mlruns'}"
        
        # Log the caller so we can clean up the tests later
        import traceback
        with open("unmocked_mlruns.log", "a") as f:
            f.write("="*60 + "\\n")
            f.write(f"WARNING: Unmocked MLflowExpManager created for {default_exp_name}!\\n")
            f.write("This means a test failed to properly mock qlib.workflow.R.\\n")
            # Only print the last 15 stack frames to avoid huge logs
            traceback.print_stack(limit=15, file=f)
            
        original_init(self, safe_uri, default_exp_name, *args, **kwargs)
        
    monkeypatch.setattr(qlib.workflow.expm.MLflowExpManager, "__init__", mocked_init)


# Add scripts directory to sys.path so bare `import env` and other script module imports work
_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'quantpits', 'scripts'))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# conftest.py can be used for globally shared fixtures

@pytest.fixture
def mock_workspace(tmp_path):
    """Fixture providing a temporary workspace directory."""
    workspace_dir = tmp_path / "MockWorkspace"
    workspace_dir.mkdir()
    return str(workspace_dir)
