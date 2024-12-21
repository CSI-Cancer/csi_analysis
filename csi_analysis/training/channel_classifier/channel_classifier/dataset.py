from pathlib import Path
import sys

from loguru import logger

# Add the project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from channel_classifier.config import (
    DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
)
from csi_images.csi_events import EventArray
from csi_utils import csi_databases

def main(
        raw_data: Path = RAW_DATA_DIR,
        interim_data: Path = INTERIM_DATA_DIR,
        processed_data: Path = PROCESSED_DATA_DIR,
        external_data: Path = EXTERNAL_DATA_DIR,
):
    logger.info(f"DATA_DIR path is: {DATA_DIR}")

    # Query the analysis table from prod database to get the list of
    # locked slides.
    slides = get_locked_slides()

    # query identifiers forthe events from ocular_hitlist from all
    # the locked slides(slide_id, frame_id, x, y)



if __name__ == "__main__":
    main()