from fastapi import APIRouter
from pydantic import BaseModel

from agents.agent_loop import run_agent

router = APIRouter(
    prefix="/agent"
)


class AskRequest(BaseModel):
    query: str


@router.post("/ask")
async def ask(request: AskRequest):
    query = request.query
    print("Initial Query in ask:-", query)
    response = await run_agent(query)
    return response
