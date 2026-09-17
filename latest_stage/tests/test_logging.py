import logging

from src.logging_utils import setup_logging


def test_setup_logging_writes_to_file(tmp_path):
    log_path = tmp_path / "run.log"
    setup_logging(log_file=log_path, level=logging.INFO, console=False)
    log = logging.getLogger("test_logger")
    log.info("hello world")
    logging.shutdown()
    contents = log_path.read_text(encoding="utf-8")
    assert "hello world" in contents
    assert "INFO" in contents


def test_setup_logging_returns_logger(tmp_path):
    log_path = tmp_path / "run.log"
    logger = setup_logging(log_file=log_path, level=logging.DEBUG, console=False, name="xyz")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "xyz"