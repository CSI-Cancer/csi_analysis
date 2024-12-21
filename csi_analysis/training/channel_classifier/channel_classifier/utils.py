

from csi_images.csi_events import EventArray
from csi_utils import csi_databases


def get_locked_slides():
    """
    Query the analysis table from prod database to get the list of
    locked slides and store in the raw data directory
    """
    pass

def get_events(slides):
    """
    Query identifiers for the events from ocular_hitlist from all
    the locked slides(slide_id, frame_id, cellx, celly, interesting,
    channel_classification)
    """
    pass

def get_event_crops(events):
    """
    Get the event crops from the csidata drive, consider multiprocessing
    Since ther are 100s of thousands of events
    """
    pass