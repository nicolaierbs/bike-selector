"""FastAPI service that receives Strava webhooks and reassigns the bike.

Strava gives the callback two seconds to answer with a 200 and retries at most
three times, so the POST handler does nothing but validate and enqueue. All the
real work (token refresh, model training, API writes) happens on a worker task.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from .config import get_settings
from .model import InsufficientData
from .service import BikeSelectorService

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("bike_selector")

service = BikeSelectorService(settings=settings)

#: Small in-memory ring buffer so `/recent` can show what the app has been doing.
RECENT: deque[dict[str, Any]] = deque(maxlen=25)


def _record(entry: dict[str, Any]) -> None:
    RECENT.appendleft({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry})


# --------------------------------------------------------------- background


def _handle_event_sync(event: dict[str, Any]) -> None:
    """Runs off the request path; must never raise into the ASGI server."""
    try:
        outcome = service.handle_event(event)
    except Exception as exc:  # noqa: BLE001 - a webhook worker must not die
        log.exception("Failed to process webhook event %s", event)
        _record({"event": event, "status": "error", "detail": str(exc)})
        return
    if outcome is None:
        log.debug("Ignoring event %s", event)
        return
    log.info("Activity %s -> %s (%s)", outcome.activity_id, outcome.status, outcome.detail)
    _record(outcome.as_dict())


async def _warmup() -> None:
    """Train once at boot so the first real ride is not stuck behind a cold model."""
    try:
        await asyncio.to_thread(service.ensure_model)
        log.info("Model ready at startup")
    except InsufficientData as exc:
        log.warning("Model not available yet: %s", exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("Warm-up training failed (will retry on first event): %s", exc)


async def _keepalive() -> None:
    """Ping our own public URL so Render's free tier does not spin the app down.

    Free instances idle out after ~15 minutes without inbound traffic, and the
    resulting cold start blows through Strava's 2 s webhook budget.
    """
    interval = settings.keepalive_minutes * 60
    if interval <= 0 or not settings.public_base_url:
        return
    url = f"{settings.public_base_url.rstrip('/')}/healthz"
    async with httpx.AsyncClient(timeout=20) as client:
        while True:
            await asyncio.sleep(interval)
            try:
                await client.get(url)
            except httpx.HTTPError as exc:
                log.debug("Keep-alive ping failed: %s", exc)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    tasks = [asyncio.create_task(_warmup()), asyncio.create_task(_keepalive())]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Strava Bike Selector", version="0.1.0", lifespan=lifespan)


# ------------------------------------------------------------------ routes


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    classifier = service.classifier
    return {
        "ok": True,
        "model_trained": classifier.is_trained,
        "model_age_hours": round(classifier.age_hours, 2) if classifier.is_trained else None,
        "bikes": [classifier.name_for(gear) for gear in classifier.labels],
    }


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "strava-bike-selector",
        "webhook_path": settings.webhook_path,
        "callback_url": settings.callback_url or None,
    }


@app.get("/model")
async def model_info() -> dict[str, Any]:
    report = service.classifier.report
    if report is None:
        return {"trained": False}
    return {
        "trained": True,
        "trained_at": report.trained_at.isoformat(),
        "n_samples": report.n_samples,
        "n_classes": report.n_classes,
        "class_counts": report.class_counts,
        "cv_accuracy": report.cv_accuracy,
        "cv_folds": report.cv_folds,
        "baseline_accuracy": report.baseline_accuracy,
        "top_features": report.top_features,
        "summary": report.summary(),
    }


@app.post("/model/retrain")
async def retrain(background: BackgroundTasks) -> dict[str, str]:
    background.add_task(service.ensure_model, force=True)
    return {"status": "retraining"}


@app.get("/recent")
async def recent() -> list[dict[str, Any]]:
    return list(RECENT)


@app.post("/activities/{activity_id}/predict")
async def predict_activity(activity_id: int, dry_run: bool = True) -> dict[str, Any]:
    """Manual trigger, handy for testing a single ride."""
    outcome = await asyncio.to_thread(
        service.process_activity, activity_id, force=True, dry_run=dry_run
    )
    return outcome.as_dict()


@app.get(settings.webhook_path)
async def verify_subscription(
    mode: str | None = Query(default=None, alias="hub.mode"),
    challenge: str | None = Query(default=None, alias="hub.challenge"),
    verify_token: str | None = Query(default=None, alias="hub.verify_token"),
) -> JSONResponse:
    """Strava's subscription validation handshake. Must answer within 2 seconds."""
    if mode != "subscribe" or verify_token != settings.webhook_verify_token:
        log.warning("Rejected webhook validation (mode=%s)", mode)
        return JSONResponse({"error": "invalid verify token"}, status_code=403)
    log.info("Webhook subscription validated")
    return JSONResponse({"hub.challenge": challenge})


@app.post(settings.webhook_path)
async def receive_event(request: Request, background: BackgroundTasks) -> JSONResponse:
    """Acknowledge immediately; do the work afterwards."""
    try:
        event = await request.json()
    except Exception:  # noqa: BLE001 - malformed body still gets a 200
        log.warning("Webhook body was not JSON")
        return JSONResponse({"status": "ignored"}, status_code=200)

    log.info(
        "Webhook: %s %s id=%s",
        event.get("aspect_type"),
        event.get("object_type"),
        event.get("object_id"),
    )
    background.add_task(_handle_event_sync, event)
    return JSONResponse({"status": "accepted"}, status_code=200)
