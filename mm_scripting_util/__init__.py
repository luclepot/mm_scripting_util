import os
import logging

from . import core

logging.getLogger("tth").addHandler(logging.NullHandler())

MODULE_PATH = os.path.dirname(__file__)
logger = logging.getLogger(__name__)
