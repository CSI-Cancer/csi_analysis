"""
This module provides a simple way to create a logger with a given level and name.
It logs in the format:
2024-10-17@14:03:46|database_examples|INFO|[message goes here]

You can also provide a file to log to, and the mode to open the file in
("a" for append, not overwriting; "w" for write).

It also provides a way to mute the logging of the pyvips module, which can be very verbose.

Example usage:
```python
import csi_logging

log = csi_logging.get_logger(level=logging.INFO, log_file="example.log")
log.info("This is an example log message.")
```
"""

import inspect
import logging
import logging.handlers

import multiprocessing
import multiprocessing.queues

import warnings

warnings.warn(
    "This module is deprecated and will be removed in a future release. "
    "Very sad since I just made it a week ago. Transition to using loguru instead.",
)

multiprocess_logging_listener = None


def get_logger(
    name="",
    level=logging.INFO,
    log_file: str = None,
    mode: str = "a",
    queue: multiprocessing.queues.Queue | logging.handlers.QueueHandler = None,
    mute_vips: bool = True,
) -> logging.Logger:
    """
    Returns a logger with the given level and name.
    :param name: by default, gets the __name__ of the calling module.
    :param level: the logging level, default is INFO.
    :param log_file: the (optional) file to log to.
    :param mode: the mode to open the file in.
    :param queue: for multiprocessing: a Queue or QueueHandler to log to.
    See https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
    :param mute_vips: whether to mute the logging of the pyvips module.
    :return: the Logger object.
    """
    # By default,
    if name == "":
        name = inspect.stack()[1].filename.split("/")[-1].split(".")[0]
    log = logging.getLogger(name)
    log.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s|%(name)s|%(levelname)s|%(message)s",
        "%Y-%m-%d@%H:%M:%S",
    )

    # Populate handlers based on args
    handlers = [logging.StreamHandler()]
    if log_file is not None and queue is not None:
        raise ValueError("Cannot log to both a file and a queue.")
    elif log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode=mode))
    elif queue is not None:
        if isinstance(queue, multiprocessing.queues.Queue):
            handlers = [logging.handlers.QueueHandler(queue)]
        elif isinstance(queue, logging.handlers.QueueHandler):
            handlers = [queue]
    # Clear handlers and then add the new ones
    log.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        log.addHandler(handler)

    if mute_vips:
        threshold_vips_logging()

    return log


def threshold_vips_logging(level=logging.WARN):
    logging.getLogger("pyvips").setLevel(level)
    logging.getLogger("pyvips.vobject").setLevel(level)
    logging.getLogger("pyvips.voperation").setLevel(level)


def start_multiprocess_logging(
    log: logging.Logger, using_ProcessPoolExecutor: bool = False
) -> multiprocessing.Queue:
    """
    Starts a process that listens for log messages in a queue and logs them.
    :param log: configured logger to match handlers for
    :param using_ProcessPoolExecutor: queue must be constructed differently if so.
    :return: the logging queue required for other processes to log to
    """
    # This should really not happen
    global multiprocess_logging_listener
    if multiprocess_logging_listener is not None:
        raise RuntimeError("Multiprocess logging already started.")
    # These have slightly different queue setups
    if using_ProcessPoolExecutor:
        queue = multiprocessing.Manager().Queue()
    else:
        queue = multiprocessing.Queue()
    # Create the QueueListener object as a global for later stopping
    multiprocess_logging_listener = logging.handlers.QueueListener(queue)
    multiprocess_logging_listener.handlers = log.handlers

    # Start the listener
    multiprocess_logging_listener.start()

    return queue


def stop_multiprocess_logging():
    """
    Stops the process that listens for log messages.
    """
    global multiprocess_logging_listener
    if isinstance(multiprocess_logging_listener, logging.handlers.QueueListener):
        multiprocess_logging_listener.enqueue_sentinel()
        multiprocess_logging_listener = None
