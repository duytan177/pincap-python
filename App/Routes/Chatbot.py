from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from App.Services.ChatbotService import ChatbotService

router = APIRouter(prefix="/api/v1", tags=["Chatbot"])


class ChatbotRequest(BaseModel):
    user_id: str
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    suggested_media_ids: Optional[List[str]] = None
    file_url: Optional[str] = None


class ChatbotResponse(BaseModel):
    intent: str
    answer: Optional[str] = None
    media: Optional[List[Dict[str, Any]]] = None
    ask_confirmation: Optional[Dict[str, Any]] = None
    action: Optional[str] = None
    album: Optional[Dict[str, Any]] = None
    frontend_link: Optional[str] = None
    error: Optional[str] = None


@router.post("/chatbot", response_model=ChatbotResponse)
async def chatbot(request: ChatbotRequest):
    """
    Main chatbot endpoint for media management queries.
    
    Handles:
    - SEARCH_MEDIA: Search and answer questions about media
    - SUGGEST_MEDIA: Suggest media and ask for album confirmation
    - CONFIRM_CREATE_ALBUM: Create album from suggested media
    - CREATE_MEDIA_FROM_INPUT: Generate metadata for new media
    - GENERAL_QA: General questions
    """
    try:
        chatbot_service = ChatbotService()
        
        response = await chatbot_service.process_message(
            user_message=request.message,
            user_id=request.user_id,
            conversation_history=request.conversation_history,
            suggested_media_ids=request.suggested_media_ids,
            file_url=request.file_url
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

