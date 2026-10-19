from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas.profile import (
    CreateProfileVersionRequest,
    ParsedFileResponse,
    ProfileIntakeChatRequest,
    ProfileIntakeChatResponse,
    ProfileIntakeMessageOut,
    ProfileSnapshot,
    ProfileSnapshotData,
    ProfileVersionSummary,
)
from app.services import profile as profile_service
from app.services import profile_intake as profile_intake_service
from app.services import resume_parsing
from app.services.stage2 import Stage2UnavailableError

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/current", response_model=ProfileSnapshot)
def get_current_profile(db: Session = Depends(get_db)):
    try:
        version = profile_service.get_current_profile(db)
    except profile_service.NoProfileError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    return version


@router.get("/versions", response_model=list[ProfileVersionSummary])
def list_profile_versions(db: Session = Depends(get_db)):
    return profile_service.list_profile_versions(db)


@router.post("", response_model=ProfileSnapshot)
def create_profile_version(payload: CreateProfileVersionRequest, db: Session = Depends(get_db)):
    """Creates a new, append-only profile version (4.4 -- the roadmap ->
    profile loop, e.g. adding a completed roadmap item's skill to
    skills.developing/core). Any Stage 2/3 result computed against an
    older version becomes stale automatically (0.1's computed staleness),
    never edited retroactively."""
    version = profile_service.create_profile_version(
        db, snapshot=payload.snapshot.model_dump(mode="json"), changelog=payload.changelog
    )
    db.commit()
    db.refresh(version)
    return version


@router.get("/intake-chat/history", response_model=list[ProfileIntakeMessageOut])
def get_intake_chat_history(db: Session = Depends(get_db)):
    """The persisted conversation, oldest first -- the desktop chat panel
    loads this on open so a prior conversation resumes."""
    return profile_intake_service.list_messages(db)


@router.delete("/intake-chat/history")
def clear_intake_chat_history(db: Session = Depends(get_db)) -> dict:
    """Starts a fresh conversation on request. Does not touch the profile
    itself, only this log."""
    profile_intake_service.clear_messages(db)
    db.commit()
    return {"cleared": True}


@router.post("/intake-chat/parse-file", response_model=ParsedFileResponse)
async def parse_intake_chat_file(file: UploadFile):
    """Extracts plain text from an uploaded resume/CV (.pdf or .docx) for
    the chat panel's "attach a resume" button -- see
    app/services/resume_parsing.py. Nothing persisted; the extracted text
    loads into the chat input for the user to review before sending."""
    content = await file.read()
    try:
        text = resume_parsing.extract_text(file.filename or "", content)
    except resume_parsing.UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    return ParsedFileResponse(text=text)


@router.post("/intake-chat", response_model=ProfileIntakeChatResponse)
def profile_intake_chat(payload: ProfileIntakeChatRequest, db: Session = Depends(get_db)):
    """One turn of the profile-intake chat (conversational alternative to
    the desktop Profile tab's structured edit form) -- see
    app/services/profile_intake.py. Persists the conversation log (both
    this message and the reply), but never writes to the profile itself,
    even when the model proposes an update -- that only happens when the
    user reviews and saves through the existing POST /profile path."""
    try:
        current_version = profile_service.get_current_profile(db)
    except profile_service.NoProfileError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    current_snapshot = ProfileSnapshotData.model_validate(current_version.snapshot)
    try:
        result = profile_intake_service.run_chat_turn(db, current_snapshot, payload.user_message)
    except Stage2UnavailableError:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured") from None
    db.commit()
    return ProfileIntakeChatResponse(**result)
