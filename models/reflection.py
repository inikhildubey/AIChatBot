from pydantic import BaseModel
from typing import Optional
from enum import Enum

from models.task import Task


class ReflectionStatus(str, Enum):
    COMPLETE = "COMPLETE"
    RETRY = "RETRY"
    WAITING = "WAITING"
    ACTION_NEEDED = "ACTION_NEEDED"


class ReflectionResponse(BaseModel):
    loop_complete: bool
    status: ReflectionStatus
    response: Optional[str] = None
    reason: Optional[str] = None
    next_task: Optional[Task] = None
