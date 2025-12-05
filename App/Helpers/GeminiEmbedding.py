# -----------------------------
# 📦 FUNCTION: GET EMBEDDING (Text or Image)
# -----------------------------
from typing import Optional
from fastapi import UploadFile
from App.Services.GeminiService import GeminiService
from typing import List

async def getEmbedding(
    text: Optional[str] = None,
    file: Optional[UploadFile] = None
) -> List[float]:
    """
    Sinh embedding cho text hoặc image.
    - Nếu có text: gọi trực tiếp Gemini Embedding API
    - Nếu có file ảnh: caption ảnh bằng GeminiService (text model) -> rồi embed caption
    """

    if not text and not file:
        raise ValueError("❌ Must provide either text or file to get embedding")

    # 🔹 Model embedding API
    gemini_embed = GeminiService(
        model="https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent",
    )

    # 1️⃣ Nếu input là TEXT → gọi embedding API trực tiếp
    if text:
        return await gemini_embed.call_gemini_api_embedding(text)

    # 2️⃣ Nếu input là IMAGE → caption trước rồi mới embed
    if file:
        caption = await getDescriptionByAi(file)
        # Gọi API embedding cho caption
        return await gemini_embed.call_gemini_api_embedding(caption)

    return []

async def getDescriptionByAi(file: Optional[UploadFile] = None, system_prompt: str = "", user_prompt: str = "") -> str:
    print("🖼️ Generating caption from image before embedding...", flush=True)
    model = "gemini-2.0-flash"
    generationConfig: dict = {
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 5000
    }

    # Dùng model text để mô tả ảnh
    gemini_text = GeminiService(
        model=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        generationConfig=generationConfig
    )

    # Default prompts nếu not pass
    default_system_prompt = """
            You are an image captioning assistant.
            
            ## TASK
            Generate a concise, factual description that captures the key visual features of the image for use in vector similarity search.
            
            ## RULES
            - Describe only visible elements (objects, people, setting, actions, colors, layout).
            - Prioritize distinctive features that help differentiate this image from others.
            - Keep it under 25 words.
            - No emotions, assumptions, or context.
            - No filler phrases like "an image of".
         """

    default_user_prompt = "Provide a short, factual description of this image."

    system_prompt = system_prompt or default_system_prompt
    user_prompt = user_prompt or default_user_prompt

    prompt = gemini_text.buildPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        files=[file]
    )
    caption = await gemini_text.textWithImageToDescription(prompt)
    return caption