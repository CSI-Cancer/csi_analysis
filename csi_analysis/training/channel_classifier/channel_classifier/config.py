from pathlib import Path
import os
import sys

from dotenv import load_dotenv
import platform

from loguru import logger
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


### Data configuration


### Model configuration