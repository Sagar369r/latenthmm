import yaml
import os
import functools

@functools.lru_cache(maxsize=1)
def get_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config = {}
    
    files_to_load = ["trading.yaml", "models.yaml", "system.yaml"]
    override_file = os.getenv("V7_CONFIG_OVERRIDE")
    if override_file:
        files_to_load.append(override_file)
        
    for filename in files_to_load:
        config_path = os.path.join(base_dir, "config", filename)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
                if data:
                    config.update(data)
                    
    return config

def load_config():
    """For backwards compatibility, clears cache and reloads."""
    get_config.cache_clear()
    return get_config()
