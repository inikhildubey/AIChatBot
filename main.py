from fastapi import FastAPI, Depends

from routes.users import router as user_router
from routes.greet import router as greet_router

app = FastAPI()


app.include_router(user_router)
app.include_router(greet_router)
#
# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
#
#
# @app.get("/world")
# def world_root(query: UserQuery = Depends()):
#     return {"message": f"Hello {query.name} & age {query.age}"}

# @app.get("/world")
# def world_root(name: str, age: int):
#     try:
#         age = int(float(age))
#     except:
#         age = 12
#
#     return {"message": f"Hello {name} & age {age}"}

# from fastapi import Query
#
# @app.get("/world")
# def world_root(
#     name: str,
#     age: int = Query(10, ge=0, le=120)
# ):
#     return {"message": f"Hello {name} & age {age}"}
