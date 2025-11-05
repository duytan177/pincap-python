from fastapi import APIRouter, HTTPException, UploadFile, Form, File
from pydantic import BaseModel
from typing import List

from App.Services.GeminiService import GeminiService

router = APIRouter(prefix="/api/v1", tags=["TextToImage"])

class TextToImageRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    files: List[UploadFile] = File(None)



@router.post("/image/generate")
async def text_to_image(
    system_prompt: str = Form(...),
    user_prompt: str = Form(...),
    files: List[UploadFile] = File(None),
):
    try:
        generationConfig: dict = {
                              "temperature": 0.9,
                              "top_p": 0.95,
                              "top_k": 40,
                              "max_output_tokens": 1024
                            }
        model = "gemini-2.0-flash-preview-image-generation"
        geminiService = GeminiService(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", generationConfig)

        prompt = geminiService.buildPrompt(system_prompt, user_prompt, files=files)
        response = await geminiService.textToImage(prompt)
        return {"data": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
