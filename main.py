import logging
import sys

from logging_config import setup_logging
from pipeline import run_pipeline

logger = logging.getLogger(__name__)


def main() -> int:
    setup_logging()
    logger.info("App started")

    try:
        run_pipeline()
        logger.info("Done")
        return 0
    except (FileNotFoundError, NotADirectoryError) as err:
        logger.error("%s", err)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())