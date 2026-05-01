from fastapi import APIRouter, Depends

from schemas.user import UserQuery

router = APIRouter(prefix="/users")


@router.get("/userprofile")
def user_profile(query: UserQuery = Depends()):
    return {"message": f"User name is {query.name} and age is {query.age}"}


@router.get("/list")
async def user_list():
    return {"message": "Success", "users": {}}


