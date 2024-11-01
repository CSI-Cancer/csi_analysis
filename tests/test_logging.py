import os
import random

from csi_analysis import csi_logging

import multiprocessing
from concurrent.futures import ProcessPoolExecutor


multiprocessing.set_start_method("spawn", force=True)


def test_logging():
    log = csi_logging.get_logger(log_file="test.log")
    log.info("This is a test log message.")
    assert os.path.isfile("test.log")
    with open("test.log", "r") as file:
        log_text = file.read()
    assert "test_logging" in log_text
    assert "This is a test log message." in log_text
    os.remove("test.log")


def dummy_worker(queue: multiprocessing.Queue = None):
    names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hank"]
    log = csi_logging.get_logger(random.choice(names), queue=queue)
    log.info("This is a test log message.")
    return True


def test_multiprocess_logging():
    log = csi_logging.get_logger(log_file="test.log")
    queue = csi_logging.start_multiprocess_logging(log)

    # Start up workers
    workers = []
    for i in range(10):
        wp = multiprocessing.Process(target=dummy_worker, args=(queue,))
        # wp = multiprocessing.Process(target=dummy_worker)
        wp.start()
        workers.append(wp)

    # Wait for workers to finish
    for worker in workers:
        worker.join()
    assert os.path.isfile("test.log")
    with open("test.log", "r") as file:
        log_text = file.read()
    assert "This is a test log message." in log_text
    csi_logging.stop_multiprocess_logging()

    log.info("One last test message.")
    assert os.path.isfile("test.log")
    with open("test.log", "r") as file:
        log_text = file.read()
    assert "One last test message." in log_text
    os.remove("test.log")


def test_processpool_logging():
    log = csi_logging.get_logger(log_file="test.log")
    queue = csi_logging.start_multiprocess_logging(log, using_ProcessPoolExecutor=True)

    # Start up workers
    with ProcessPoolExecutor() as executor:
        for i in range(10):
            executor.submit(dummy_worker, queue)

    assert os.path.isfile("test.log")
    with open("test.log", "r") as file:
        log_text = file.read()
    assert "This is a test log message." in log_text
    current_length = len(log_text)

    # Do it again
    # Start up workers
    with ProcessPoolExecutor() as executor:
        for i in range(10):
            executor.submit(dummy_worker, queue)

    assert os.path.isfile("test.log")
    with open("test.log", "r") as file:
        log_text = file.read()
    assert "This is a test log message." in log_text
    assert len(log_text) > current_length
    os.remove("test.log")
    csi_logging.stop_multiprocess_logging()
