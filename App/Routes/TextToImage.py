from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, Form, File

from App.Services.CFWorkerService import CFWorkerService
import os

router = APIRouter(prefix="/api/v1", tags=["TextToImage"])

CLOUDFLARE_WORKER_URL = os.getenv("CLOUDFLARE_WORKER_PINCAP")

@router.post("/image/generate")
async def text_to_image(
    system_prompt: Optional[str] = Form(None),
    user_prompt: str = Form(...),
    size: str = Form("512x512"),
    files: List[UploadFile] = File(None),
):
    try:
        # 🔹 Parse size "WxH"
        size_parts = size.split("x")
        width = int(size_parts[0]) if len(size_parts) > 0 else 512
        height = int(size_parts[1]) if len(size_parts) > 1 else 512

        # 🔹 Prepare CFWorkerService
        cf_service = CFWorkerService(CLOUDFLARE_WORKER_URL)

        # 🔹 Read first file bytes if exists
        file = files[0] if files else None
        # 🔹 Call CF Worker
        img_base64 = await cf_service.generate_image(
            prompt=user_prompt,
            action="generate",
            image_file=file,
            width=width,
            height=height,
        )

        return {"data": img_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
