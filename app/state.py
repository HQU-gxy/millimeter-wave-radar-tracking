from pydantic import BaseModel
from typing import Iterable, Literal, Optional, Sequence
from enum import Enum, auto
from capture.model import END_MAGIC, Target, Targets
from datetime import datetime, timedelta


class ArbiterResult(Enum):
    """
    The result of the arbiter
    """

    INDECISIVE = 0x0
    """
    reserved
    """
    MOVING = 1
    """
    object present and moving actively
    """
    STILL = 2
    """
    object present but not moving actively
    """
    IDLE = 3
    """
    no object present
    """


class DoorState(Enum):
    OPEN = auto()
    CLOSED = auto()


class DoorSignal(Enum):
    UP = auto()
    DOWN = auto()


class MaybeTarget(BaseModel, frozen=True):
    target: Optional[Target]
    timestamp: datetime
