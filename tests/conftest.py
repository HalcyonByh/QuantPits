import pytest
import os
import sys

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
