import os
from pathlib import Path
from dotenv import load_dotenv

# Find .env file and load it
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Load configuration values
_ecgfm_path_str = os.getenv("ECGFM_FT_WEIGHT_PATH")
if not _ecgfm_path_str:
    raise ValueError("ECGFM_FT_WEIGHT_PATH environment variable is not set")

ECGFM_FT_WEIGHT_PATH = Path(_ecgfm_path_str).expanduser()
