import numpy as np
import os
import random as rn
import torch
import logging
import yaml

def seed_everything(seed=1234):
    rn.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logging(log_file_path):
    """Sets up logging to a file."""
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    LOG_FORMAT = '%(asctime)s [%(name)s] %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S %a')
    logger.addHandler(fh)    
    return logger

def load_yaml(filepath):
    """Loads a YAML file and returns the data as a Python object."""
    try:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)  # Use safe_load to prevent arbitrary code execution
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML: {e}")
        return None
