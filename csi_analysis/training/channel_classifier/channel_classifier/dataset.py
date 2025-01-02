from pathlib import Path
import sys

from loguru import logger

#ML related imports
import pandas as pd

# Add the project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from channel_classifier.config import (
    DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
)

from channel_classifier.utils import (
    get_events
)

def main(
        raw_data: Path = RAW_DATA_DIR,
        interim_data: Path = INTERIM_DATA_DIR,
        processed_data: Path = PROCESSED_DATA_DIR,
        external_data: Path = EXTERNAL_DATA_DIR,
):
    logger.info(f"DATA_DIR path is: {DATA_DIR}")

    """
    # Query the analysis table from prod database to get the list of
    # locked slides and store in the raw data directory
    """
    slides = pd.read_csv(raw_data / "locked_slides.csv")
    slides = slides["slide_id"].unique()
    """
    # query identifiers for the events from ocular_hitlist from all
    # the locked slides(slide_id, frame_id, cellx, celly, interesting,
    # channel_classification)
    """
    events, labels = get_events(slides)
    if len(events) != len(labels):
        assert False, "Events and labels are not of the same length"
        
    """
    # Get the event crops from the csidata drive, consider multiprocessing
    # Since ther are 100s of thousands of events
    """
    # events = get_event_crops(events)

    # store in the interim data directory

if __name__ == "__main__":
    main()