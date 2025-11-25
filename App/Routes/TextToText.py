from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from App.Services.GeminiService import GeminiService

router = APIRouter(prefix="/api/v1", tags=["TextToText"])


class TextToTextRequest(BaseModel):
    system_prompt: str
    user_prompt: str

@router.post("/text-to-text")
async def text_to_text(request: TextToTextRequest):
    try:
        generationConfig: dict = {
                              "temperature": 0.9,
                              "top_p": 0.95,
                              "top_k": 40,
                              "max_output_tokens": 1024,
                              "responseModalities": ["TEXT"]

        }
        model = "gemini-2.5-flash"
        geminiService = GeminiService(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", generationConfig)
        system_prompt = request.system_prompt
        user_prompt = request.user_prompt
        prompt = geminiService.buildPrompt(system_prompt, user_prompt)

        response = await geminiService.textToText(prompt)
        return {"data": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
