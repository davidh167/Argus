"""Seed (or add a new version of) the profile from a YAML file.

Usage:
    poetry run python scripts/seed_profile.py [path/to/profile.yaml]

Defaults to profile_seed.yaml at the repo root. Profile versions are
append-only -- running this again after editing the YAML creates a NEW
profile_versions row rather than overwriting the last one.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import SessionLocal
from app.schemas.profile import ProfileSnapshotData
from app.services.profile import create_profile_version
from app.services.relevance import rescore_all


def main() -> None:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[1] / "profile_seed.yaml"
    )
    raw = yaml.safe_load(path.read_text())

    changelog = raw.pop("changelog", None)
    # Validate against the schema before writing -- fails loudly on a
    # missing required field. A typo'd/unrecognized field name does NOT
    # fail: Pydantic's default is to silently drop unknown keys, not
    # reject them, so double-check the printed snapshot below actually
    # has what you meant to set.
    snapshot = ProfileSnapshotData.model_validate(raw).model_dump(mode="json")

    with SessionLocal() as db:
        version = create_profile_version(db, snapshot=snapshot, changelog=changelog)
        # A new profile version changes what "relevant" means -- rescore
        # every posting against it now rather than leaving relevance_score
        # stale until the next crawl/scrape/hydrate happens to trigger it.
        rescore_result = rescore_all(db)
        db.commit()
        print(f"Created profile_versions id={version.id} (changelog: {changelog!r})")
        print(f"Rescored {rescore_result['rescored']} postings against it.")


if __name__ == "__main__":
    main()
