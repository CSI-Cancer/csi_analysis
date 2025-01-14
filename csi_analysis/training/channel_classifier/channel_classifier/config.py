from pathlib import Path
import os
import sys

from dotenv import load_dotenv
import platform

from loguru import logger
import yaml
from types import SimpleNamespace
load_dotenv()


### System configuration

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Local data storage location
if PROJ_ROOT:
    os.makedirs(PROJ_ROOT/"data", exist_ok=True)
    DATA_DIR = PROJ_ROOT / "data"
else:
    logger.error("PROJ_ROOT is not defined")
    sys.exit(1)

# Network data storage location
if platform.system() == "Windows":
    NETWORK_ROOT = "\\\\csi-nas.usc.edu"
elif platform.system() == "Linux":
    NETWORK_ROOT = "/mnt"
else:
    logger.info("Unknown operating system")

# Local repo data storage location
RAW_DATA_DIR = DATA_DIR / "raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)
INTERIM_DATA_DIR = DATA_DIR / "interim"
os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
PROCESSED_DATA_DIR = DATA_DIR / "processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
EXTERNAL_DATA_DIR = DATA_DIR / "external"
os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)

# Trained model storage location
MODELS_DIR = PROJ_ROOT / "models"

# Analysis and reports
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Hyper parameter tuning config
tune_config = {
    "wandb_key": "f50e7404c274fd0240b3de443ad368d762b16643",
    "tune": True,
    "count" : 20,
    "augment" : False,
    "seed" : 42,
    "debug" : False,
    "device" : "cuda:0",
}

# Sweep config
with open(PROJ_ROOT / "channel_classifier" / "sweep_config.yml") as f:
    sweep_config = yaml.safe_load(f)

### Data configuration
class_map = {
    'D':0,
    'CK':1,
    'CD':2,
    'V':3,
    'CK|CD|V':4,
    'CK|CD':5,
    'D|CK|CD|V':6,
    'CK|V':7,
    'D|CK|CD':8,
    'D|CK|V':9,
    'D|V':10,
    'D|CD|V':11,
    'D|CD':12,
    'D|CK':13,
    'CD|V':14,
}

pred_encoder = {
    0: 'D',
    1: 'CK',
    2: 'CD',
    3: 'V',
    4: 'CK|CD|V',
    5: 'CK|CD',
    6: 'D|CK|CD|V',
    7: 'CK|V',
    8: 'D|CK|CD',
    9: 'D|CK|V',
    10: 'D|V',
    11: 'D|CD|V',
    12: 'D|CD',
    13: 'D|CK',
    14: 'CD|V',
}

# Test sweep configurations
test_sweep = {
    "split": "test",
    "batch_size": 728,
    "device": "cuda:0",
    "dropout": 0.5,
    "num_classes": 15,
    "model": 'generic',
    "model_path": MODELS_DIR / "saved" / "best_checkpoint_70.pth",
    "top_k": 10,
}
test_sweep = SimpleNamespace(**test_sweep)