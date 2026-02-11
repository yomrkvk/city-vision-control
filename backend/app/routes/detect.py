from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.stub_detector import StubDetector

router = APIRouter(prefix="/detect", tags=["detect"])

detector = StubDetector()


@router.post("/")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    result = detector.predict(image_bytes)

    return result
