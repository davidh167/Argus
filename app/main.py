from fastapi import Depends, FastAPI

from app.auth import verify_token
from app.routers import applications, batches, companies, postings, profile, roadmap

app = FastAPI(title="Argus", version="0.1.0")

app.include_router(postings.router, dependencies=[Depends(verify_token)])
app.include_router(profile.router, dependencies=[Depends(verify_token)])
app.include_router(companies.router, dependencies=[Depends(verify_token)])
app.include_router(applications.router, dependencies=[Depends(verify_token)])
app.include_router(roadmap.router, dependencies=[Depends(verify_token)])
app.include_router(batches.router, dependencies=[Depends(verify_token)])


@app.get("/health")
def health():
    return {"status": "ok"}
