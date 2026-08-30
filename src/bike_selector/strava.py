"""Thin wrapper around stravalib: token lifecycle, gear lookup, activity read/write."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx
from stravalib import Client

from .config import Settings, get_settings

log = logging.getLogger(__name__)

PUSH_SUBSCRIPTION_URL = "https://www.strava.com/api/v3/push_subscriptions"

#: Sport types that ride on a bike and therefore accept a bike as gear.
#: Restricted to exactly "Ride" — VirtualRide is excluded because virtual rides
#: are always done on the same (trainer) bike, so it carries no signal and would
#: only add noise; the other Ride subtypes (GravelRide, MountainBikeRide, etc.)
#: are excluded too so the eligible set is exactly sport_type "Ride".
BIKE_SPORT_TYPES: tuple[str, ...] = ("Ride",)


class StravaAuthError(RuntimeError):
    """Raised when we cannot obtain a usable access token."""


@dataclass(frozen=True)
class Bike:
    id: str
    name: str
    primary: bool = False
    retired: bool = False
    distance_m: float = 0.0


def sport_type_of(activity: Any) -> str:
    """Return the sport type as a plain string across stravalib model variants."""
    for attr in ("sport_type", "type"):
        value = getattr(activity, attr, None)
        if value is None:
            continue
        # stravalib 2.x wraps these in RelaxedSportType / RelaxedActivityType root models.
        root = getattr(value, "root", value)
        text = getattr(root, "value", root)
        text = str(text).strip()
        if text and text.lower() not in {"none", "nan"}:
            return text
    return "Unknown"


class StravaGateway:
    """Owns the Strava client and keeps its access token fresh.

    The refresh token comes from the environment. Strava may hand back a rotated
    refresh token, so the newest one is cached in the (ephemeral) state dir and a
    warning is logged telling you to update the environment variable.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self._client = Client(rate_limit_requests=True)
        self._access_token: str | None = None
        self._expires_at: float = 0.0
        self._refresh_token: str = self._load_refresh_token()
        self._athlete_id: int | None = None

    # ------------------------------------------------------------------ auth

    def _load_refresh_token(self) -> str:
        """Prefer a cached rotated token, fall back to the configured one."""
        path = self.settings.token_path
        if path.exists():
            try:
                cached = json.loads(path.read_text())
                token = cached.get("refresh_token")
                if token:
                    return str(token)
            except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
                log.warning("Ignoring unreadable token cache %s: %s", path, exc)
        return self.settings.strava_refresh_token

    def _persist_refresh_token(self, token: str) -> None:
        if token == self._refresh_token:
            return
        log.warning(
            "Strava rotated the refresh token. Update STRAVA_REFRESH_TOKEN in your "
            "environment to %s… or the app will stop working after the next restart.",
            token[:8],
        )
        self._refresh_token = token
        try:
            self.settings.ensure_state_dir()
            self.settings.token_path.write_text(json.dumps({"refresh_token": token}))
        except OSError as exc:  # pragma: no cover - defensive
            log.warning("Could not cache refresh token: %s", exc)

    @property
    def client(self) -> Client:
        """A stravalib client with a valid access token."""
        with self._lock:
            if self._access_token is None or time.time() > self._expires_at - 120:
                self._refresh()
            return self._client

    def _refresh(self) -> None:
        settings = self.settings
        if not (settings.strava_client_id and settings.strava_client_secret):
            raise StravaAuthError("STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET must be set.")
        if not self._refresh_token:
            raise StravaAuthError(
                "No refresh token. Run `uv run bike-selector auth` and set STRAVA_REFRESH_TOKEN."
            )

        log.info("Refreshing Strava access token")
        info = self._client.refresh_access_token(
            client_id=settings.strava_client_id,
            client_secret=settings.strava_client_secret,
            refresh_token=self._refresh_token,
        )
        self._access_token = info["access_token"]
        self._expires_at = float(info["expires_at"])
        self._client.access_token = self._access_token
        self._persist_refresh_token(str(info["refresh_token"]))

    # ------------------------------------------------------------- read side

    def athlete_id(self) -> int:
        if self._athlete_id is None:
            self._athlete_id = int(self.client.get_athlete().id)
        return self._athlete_id

    def bikes(self) -> list[Bike]:
        athlete = self.client.get_athlete()
        bikes: list[Bike] = []
        for gear in athlete.bikes or []:
            gear_id = getattr(gear, "id", None)
            if not gear_id:
                continue
            bikes.append(
                Bike(
                    id=str(gear_id),
                    name=str(getattr(gear, "name", None) or gear_id),
                    primary=bool(getattr(gear, "primary", False)),
                    retired=bool(getattr(gear, "retired", False)),
                    distance_m=float(getattr(gear, "distance", 0) or 0),
                )
            )
        return bikes

    def bike_names(self) -> dict[str, str]:
        return {bike.id: bike.name for bike in self.bikes()}

    def recent_activities(self, limit: int | None = None) -> list[Any]:
        """Summary activities, newest first.

        The summary payload already carries every feature the model needs
        (including ``gear_id``), so training costs one paginated call rather than
        one request per activity.
        """
        limit = limit or self.settings.training_activity_limit
        return list(self.client.get_activities(limit=limit))

    def activity(self, activity_id: int) -> Any:
        return self.client.get_activity(activity_id)

    # ------------------------------------------------------------ write side

    def update_activity(
        self,
        activity_id: int,
        *,
        gear_id: str | None = None,
        description: str | None = None,
        name: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if gear_id is not None:
            kwargs["gear_id"] = gear_id
        if description is not None:
            kwargs["description"] = description
        if name is not None:
            kwargs["name"] = name
        if not kwargs:
            return None
        log.info("Updating activity %s: %s", activity_id, ", ".join(sorted(kwargs)))
        return self.client.update_activity(activity_id, **kwargs)

    # ---------------------------------------------------- webhook management

    def _subscription_auth(self) -> dict[str, str]:
        return {
            "client_id": str(self.settings.strava_client_id),
            "client_secret": self.settings.strava_client_secret,
        }

    def list_subscriptions(self) -> list[dict[str, Any]]:
        response = httpx.get(PUSH_SUBSCRIPTION_URL, params=self._subscription_auth(), timeout=30)
        response.raise_for_status()
        return response.json()

    def create_subscription(self, callback_url: str, verify_token: str) -> dict[str, Any]:
        response = httpx.post(
            PUSH_SUBSCRIPTION_URL,
            data={
                **self._subscription_auth(),
                "callback_url": callback_url,
                "verify_token": verify_token,
            },
            timeout=30,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Strava rejected the subscription: {response.text}")
        return response.json()

    def delete_subscription(self, subscription_id: int) -> None:
        response = httpx.delete(
            f"{PUSH_SUBSCRIPTION_URL}/{subscription_id}",
            params=self._subscription_auth(),
            timeout=30,
        )
        if response.status_code not in (200, 204):
            raise RuntimeError(f"Delete failed: {response.status_code} {response.text}")
