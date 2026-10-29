import logging

import anthropic
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.auth import verify_token
from app.routers import applications, batches, companies, postings, profile, roadmap

logger = logging.getLogger(__name__)


async def anthropic_auth_error_handler(request: Request, exc: Exception):
    """Anthropic rejecting the configured key (revoked or expired) used to surface as an opaque
    500. Say what actually happened, and log it. 502 rather than 401 so it can't be mistaken for
    this API's own token check."""
    logger.error(
        "Anthropic API rejected the configured key on %s %s (request_id=%s)",
        request.method,
        request.url.path,
        getattr(exc, "request_id", None),
    )
    return JSONResponse(
        status_code=502,
        content={
            "detail": "Anthropic rejected ANTHROPIC_API_KEY (401). Check that the key in .env is "
            "valid, then restart the backend."
        },
    )


app = FastAPI(title="Argus", version="0.1.0")
app.add_exception_handler(anthropic.AuthenticationError, anthropic_auth_error_handler)

app.include_router(postings.router, dependencies=[Depends(verify_token)])
app.include_router(profile.router, dependencies=[Depends(verify_token)])
app.include_router(companies.router, dependencies=[Depends(verify_token)])
app.include_router(applications.router, dependencies=[Depends(verify_token)])
app.include_router(roadmap.router, dependencies=[Depends(verify_token)])
app.include_router(batches.router, dependencies=[Depends(verify_token)])


@app.get("/health")
def health():
    return {"status": "ok"}
