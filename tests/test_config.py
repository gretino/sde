import os
import pytest
from pathlib import Path

def test_config_loads_ecgfm_weight_path(monkeypatch):
    # Setup the environment variable
    fake_path = "~/fake/path/to/weights.pt"
    monkeypatch.setenv("ECGFM_FT_WEIGHT_PATH", fake_path)
    
    # We need to reload the config module to pick up the new env var if it was already loaded
    import sde.config
    import importlib
    importlib.reload(sde.config)
    
    # Verify the path is correctly resolved
    expected_path = Path(fake_path).expanduser()
    assert sde.config.ECGFM_FT_WEIGHT_PATH == expected_path

def test_config_raises_if_weight_path_missing(monkeypatch):
    monkeypatch.delenv("ECGFM_FT_WEIGHT_PATH", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: None)
    
    with pytest.raises(ValueError, match="ECGFM_FT_WEIGHT_PATH environment variable is not set"):
        import sde.config
        import importlib
        importlib.reload(sde.config)
