from fastapi import APIRouter, UploadFile, File
from tools.rag_tool import upload_data, ask_question

router = APIRouter(prefix="/greet")


@router.get("/")
def greet_welcome(name: str):
    return {"message": f"Welcome to the Greet Module {name}"}


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    return await upload_data(file)


@router.post("/ask")
async def ask(query: str):
    response = await ask_question(query)
    return response
