from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.profile_version import ProfileVersion


class NoProfileError(RuntimeError):
    pass


def get_current_profile(db: Session) -> ProfileVersion:
    """Latest profile_versions row by id (versions are append-only, never
    edited in place -- spec section 4.1)."""
    version = db.scalar(select(ProfileVersion).order_by(ProfileVersion.id.desc()).limit(1))
    if version is None:
        raise NoProfileError(
            "No profile_versions row exists yet -- run scripts/seed_profile.py first."
        )
    return version


def create_profile_version(
    db: Session, snapshot: dict, changelog: str | None = None
) -> ProfileVersion:
    version = ProfileVersion(snapshot=snapshot, changelog=changelog)
    db.add(version)
    db.flush()
    return version


def list_profile_versions(db: Session) -> list[ProfileVersion]:
    """Newest first -- the Profile tab's version history (3.9, read-only)."""
    return list(db.scalars(select(ProfileVersion).order_by(ProfileVersion.id.desc())))
