from pydantic import BaseModel
from typing import Iterable, Literal, Optional, Sequence
from enum import Enum, auto
from capture.model import END_MAGIC, Target, Targets
from datetime import datetime, timedelta

class ArbiterResult(Enum):
    """
    The result of the arbiter
    """
    INDECISIVE = auto()
    """
    reserved
    """
    MOVING = auto()
    """
    object present and moving actively
    """
    STILL = auto()
    """
    object present but not moving actively
    """
    IDLE = auto()
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
