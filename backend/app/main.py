from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.detect import router as detect_router

app = FastAPI(title="City Vision Control API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect_router)


@app.get("/")
def healthcheck():
    return {"status": "ok"}

