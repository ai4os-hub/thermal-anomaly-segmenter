"""Module for defining custom API response parsers and content types.
This module is used by the API server to convert the output of the requested
method into the desired format.

The module shows simple but efficient example functions. However, you may
need to modify them for your needs.
"""
import logging
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def json_response(result, **options):
    """Converts the prediction results into json return format.

    Arguments:
        result -- Result value from call, expected either dict or str
        options -- Not used, added for illustration purpose.

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        Converted result into json dictionary format.
    """
    logger.debug("Response result type: %d", type(result))
    logger.debug("Response result: %d", result)
    logger.debug("Response options: %d", options)
    try:
        if isinstance(result, (dict, list, str)):
            return result
        # if isinstance(result, np.ndarray):
        #     return result.tolist()
        return dict(result)
    except Exception as err:  # TODO: Fix to specific exception
        logger.warning("Error converting result to json: %s", err)
        raise RuntimeError("Unsupported response type") from err


def png_response(result, **options):
    """Converts the prediction results into a image png return format.

    Arguments:
        result -- Result value from call, expected either dict or str
        options -- Not used, added for illustration purpose.

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        Converted result into json dictionary format.
    """
    logger.debug("Response result type: %d", type(result))
    logger.debug("Response result: %d", result)
    logger.debug("Response options: %d", options)
    result = result['result']
    try:
        if Path(result).is_file():
            return open(result, "rb")
        else:
            raise Exception
    except Exception as err:  # TODO: Fix to specific exception
        logger.error("Error converting result to png: %s", err)


content_types = {
    "application/json": json_response,
    "image/png": png_response,
}
