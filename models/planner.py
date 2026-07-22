from pydantic import BaseModel
from models.task import Task


class PlannerResponse(BaseModel):
    tasks: list[Task]