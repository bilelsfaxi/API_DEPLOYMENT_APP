from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
import os
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from .. import crud, models, schemas
from ..database import get_db
from fastapi.responses import RedirectResponse

def create_router(templates_directory: str) -> APIRouter:
    """Crée et configure le routeur pour l'interface utilisateur."""
    router = APIRouter(prefix="/ui", tags=["UI"])
    templates = Jinja2Templates(directory=templates_directory)

    @router.get("/session/{session_id}", response_class=HTMLResponse)
    async def session_ui(request: Request, session_id: int, db: AsyncSession = Depends(get_db)):
        """Route pour afficher l'interface de la session avec les vidéos de référence."""
        session = await crud.get_session_by_id(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")
        
        videos = await crud.get_next_videos_for_session(db, session_id)
        # Convertir les objets Pydantic VideoReference en dictionnaires pour la sérialisation JSON dans Jinja2
        videos_for_template = [video.model_dump(mode="json") for video in videos]
        
        # Récupérer toutes les postures possibles pour le menu déroulant
        all_postures = [p.value for p in models.PostureEnum]

        context = {
            "request": request,
            "session_id": session_id,
            "dog_id": session.dog_id,
            "posture": session.posture,
            "videos": videos_for_template,
            "all_postures": all_postures
        }
        
        html_content = templates.get_template("session_ui.html").render(context)
        return HTMLResponse(content=html_content)

    @router.post("/create-new-session", response_class=RedirectResponse)
    async def create_new_session(request: Request, db: AsyncSession = Depends(get_db)):
        """Crée une nouvelle session et redirige vers la page de cette session."""
        form_data = await request.form()
        dog_id_str = form_data.get("dog_id")
        posture_str = form_data.get("posture")
        
        if not dog_id_str or not posture_str:
            raise HTTPException(status_code=400, detail="Dog ID and posture are required.")

        dog_id = int(dog_id_str)
        new_session_data = schemas.db_schemas.VideoSessionCreate(dog_id=dog_id, posture=models.PostureEnum(posture_str))
        new_session = await crud.create_video_session(db=db, session=new_session_data)
        
        return RedirectResponse(url=f"/ui/session/{new_session.id}", status_code=303)

    return router