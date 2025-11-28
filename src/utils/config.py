import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Load and validate YAML configs"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        
    def __getattr__(self, key):
        value = self._config.get(key)
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def __getitem__(self, key):
        return self._config[key]
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def to_dict(self):
        return self._config

def load_config(config_path: str) -> Config:
    """Load YAML config file"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)
