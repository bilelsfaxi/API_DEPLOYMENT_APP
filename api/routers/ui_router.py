from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from .. import crud
from ..database import get_db

router = APIRouter(prefix="/ui", tags=["UI"])

templates = Jinja2Templates(directory="api/templates")

@router.get("/session/{session_id}", response_class=HTMLResponse)
async def session_ui(request: Request, session_id: int, db: AsyncSession = Depends(get_db)):
    """Route pour afficher l'interface de la session avec les vidéos de référence."""
    session = await crud.get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")
    
    videos = await crud.get_next_videos_for_session(db, session_id)
    # Convertir les objets Pydantic VideoReference en dictionnaires pour la sérialisation JSON dans Jinja2
    videos_for_template = [video.model_dump(mode="json") for video in videos]
    
    context = {
        "request": request,
        "session_id": session_id,
        "posture": session.posture,
        "videos": videos_for_template
    }
    
    html_content = templates.get_template("session_ui.html").render(context)
    return HTMLResponse(content=html_content)