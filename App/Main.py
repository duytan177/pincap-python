from fastapi import FastAPI
from App.Routes import TextToText, TextToImage, SearchByMedia, GenerateMediaMetadata, SearchMediaByTextEmbedding, Chatbot

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(title="Gemini AI FastAPI Service")


app.include_router(TextToText.router)
app.include_router(TextToImage.router)
app.include_router(SearchByMedia.router)
app.include_router(GenerateMediaMetadata.router)
app.include_router(SearchMediaByTextEmbedding.router)
app.include_router(Chatbot.router)
@app.get("/")
def root():
    return {"message": "Gemini AI FastAPI Service is running"}
