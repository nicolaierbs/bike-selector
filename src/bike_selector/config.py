"""Configuration, loaded from the environment (and a local .env in development)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_` is a protected prefix in pydantic; opt out so MODEL_TTL_HOURS works.
        protected_namespaces=(),
    )

    # --- Strava application credentials ---
    strava_client_id: int = 0
    strava_client_secret: str = ""
    strava_refresh_token: str = ""

    # --- Webhook ---
    webhook_verify_token: str = "change-me"
    webhook_path_secret: str = ""
    public_base_url: str = ""

    # --- Behaviour ---
    set_gear: bool = True
    write_description: bool = True
    overwrite_existing_gear: bool = True
    min_confidence: float = 0.0
    skip_trainer: bool = False
    description_marker: str = "[bike-selector]"
    max_probabilities_shown: int = 3
    info_url: str = "https://www.erbs.eu/bikeselector/"

    # --- Title ---
    rename_title: bool = True
    #: Replace even a title the rider typed themselves, not just Strava's
    #: generic "Morning Ride" default.
    overwrite_existing_title: bool = False
    #: Maps a bike's Strava gear_id to the German category word used in the
    #: title, e.g. {"b111": "Rennrad", "b222": "Gravel"}. A bike missing from
    #: this map falls back to its own Strava name.
    bike_types: dict[str, str] = Field(default_factory=dict)
    #: How many of the ride's longest climb segments to name in the title.
    climb_segments_in_title: int = 2
    #: Minimum average grade (%) for a segment effort to count as a climb.
    climb_min_grade: float = 3.0

    # --- Model ---
    training_activity_limit: int = 1000
    model_ttl_hours: float = 24.0
    recency_half_life_days: float = 365.0
    min_rides_per_bike: int = 5

    # --- Runtime ---
    state_dir: Path = Path("/tmp/bike-selector")
    keepalive_minutes: int = 10
    log_level: str = "INFO"

    # --- OAuth helper (local `bike-selector auth` flow) ---
    oauth_local_port: int = Field(default=8721, description="Port for the local auth callback.")

    @property
    def webhook_path(self) -> str:
        """Path the Strava webhook posts to, optionally hardened with a secret slug."""
        if self.webhook_path_secret:
            return f"/webhook/{self.webhook_path_secret}"
        return "/webhook"

    @property
    def callback_url(self) -> str:
        return f"{self.public_base_url.rstrip('/')}{self.webhook_path}"

    @property
    def model_path(self) -> Path:
        return self.state_dir / "model.pkl"

    @property
    def token_path(self) -> Path:
        return self.state_dir / "token.json"

    @property
    def auto_labelled_path(self) -> Path:
        return self.state_dir / "auto_labelled.json"

    def ensure_state_dir(self) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        return self.state_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
