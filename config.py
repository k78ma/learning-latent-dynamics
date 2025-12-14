import os
import importlib
from typing import Any

def _load_config(module_path: str) -> Any:
    module = importlib.import_module(module_path)
    if not hasattr(module, "e2c_config"):
        raise AttributeError(f"Module '{module_path}' does not define 'e2c_config'")
    return getattr(module, "e2c_config")


_DEFAULT_MODULE = os.environ.get("E2C_CONFIG", "configs.base_config")
e2c_config = _load_config(_DEFAULT_MODULE)

