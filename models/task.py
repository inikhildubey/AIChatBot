#models/task.py

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    tool: str
    query: str
    priority: int = 1
    retry_count: int = 0
    depends_on: Optional[list[str]] = Field(default_factory=list)
