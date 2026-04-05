from typing import Union

from pydantic import BaseModel, field_validator


class UserQuery(BaseModel):
    name: str
    age: Union[int, float, str]

    @field_validator("age", mode="before")
    def handle_age(cls, value):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 12
