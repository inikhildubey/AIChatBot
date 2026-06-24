from fastapi import APIRouter

from agents.agent_loop import run_agent

router = APIRouter(
    prefix="/agent"
)


@router.post("/ask")
async def ask(query: str):
    response = await run_agent(query)
    return response
