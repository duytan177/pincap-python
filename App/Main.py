from fastapi import FastAPI
from App.Routes import TextToText, TextToImage, SearchByMedia

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(title="Gemini AI FastAPI Service")


app.include_router(TextToText.router)
app.include_router(TextToImage.router)
app.include_router(SearchByMedia.router)


@app.get("/")
def root():
    return {"message": "Gemini AI FastAPI Service is running"}
