#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from enum import Enum


class TransformersType(Enum):
    NONE = 0
    LOG = 1
    BOX_COX = 2
    SQRT = 3
