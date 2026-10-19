from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas.roadmap import (
    DemandCluster,
    GenerateRoadmapBriefRequest,
    ReadinessView,
    SkillJudgmentPreview,
    SkillJudgmentRun,
    SkillRoadmapOut,
    SkillTrend,
    UpdateRoadmapStatusRequest,
)
from app.services import roadmap as roadmap_service
from app.services import skill_judgment
from app.services.stage2 import Stage2UnavailableError

router = APIRouter(prefix="/roadmap", tags=["roadmap"])


@router.get("/trends", response_model=list[SkillTrend])
def get_skill_trends(
    threshold: int = 3,
    posting_ids: list[int] | None = Query(None),
    db: Session = Depends(get_db),
):
    return roadmap_service.detect_skill_trends(db, threshold=threshold, posting_ids=posting_ids)


@router.get("/readiness", response_model=ReadinessView)
def get_readiness_view(db: Session = Depends(get_db)):
    return roadmap_service.readiness_view(db)


@router.get("/demand", response_model=list[DemandCluster])
def get_market_demand(
    posting_ids: list[int] | None = Query(None),
    sort: Literal["market", "interest"] = "market",
    db: Session = Depends(get_db),
):
    return roadmap_service.market_demand(db, posting_ids=posting_ids, sort=sort)


@router.get("/judge-skills/preview", response_model=SkillJudgmentPreview)
def preview_skill_judgments(
    limit: int = skill_judgment.DEFAULT_LIMIT, db: Session = Depends(get_db)
):
    return skill_judgment.preview(db, limit=limit)


@router.post("/judge-skills", response_model=SkillJudgmentRun)
def judge_skills(limit: int = skill_judgment.DEFAULT_LIMIT, db: Session = Depends(get_db)):
    try:
        return skill_judgment.judge_pending(db, limit=limit)
    except Stage2UnavailableError:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured") from None


@router.post("/brief", response_model=SkillRoadmapOut)
def generate_brief(payload: GenerateRoadmapBriefRequest, db: Session = Depends(get_db)):
    try:
        roadmap = roadmap_service.generate_roadmap_brief(
            db,
            payload.skill_or_language,
            trigger_type=payload.trigger_type,
            trend_evidence=payload.trend_evidence,
        )
    except Stage2UnavailableError:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured") from None
    db.commit()
    db.refresh(roadmap)
    return roadmap


@router.get("", response_model=list[SkillRoadmapOut])
def list_roadmap(status: str | None = None, db: Session = Depends(get_db)):
    return roadmap_service.list_roadmap(db, status=status)


@router.patch("/{roadmap_id}/status", response_model=SkillRoadmapOut)
def update_status(
    roadmap_id: int, payload: UpdateRoadmapStatusRequest, db: Session = Depends(get_db)
):
    roadmap = roadmap_service.update_roadmap_status(db, roadmap_id, payload.status)
    if roadmap is None:
        raise HTTPException(status_code=404, detail="Roadmap item not found")
    db.commit()
    db.refresh(roadmap)
    return roadmap
