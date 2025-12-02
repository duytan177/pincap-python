from fastapi import APIRouter
from pydantic import BaseModel
import os
from App.Services.CFWorkerService import CFWorkerService
from asyncio import run

router = APIRouter(prefix="/api/v1", tags=["video"])

class VideoRequest(BaseModel):
    video_url: str

detect_service = CFWorkerService(
    worker_url=os.getenv("CLOUDFLARE_WORKER_PINCAP_DETECT_VIDEO")
)

@router.post("/detect-video")
def detect_video(req: VideoRequest):
    return run(detect_service.extract_and_describe(req.video_url))
