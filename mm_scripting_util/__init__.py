import os
import logging

import mm_scripting_util.core

logging.getLogger("mm_scripting_util").addHandler(logging.NullHandler())

MODULE_PATH = os.path.dirname(__file__)
logger = logging.getLogger(__name__)
