"""Orchestration: webhook event -> prediction -> Strava update."""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import Settings, get_settings
from .features import gear_id_of, is_bike_activity
from .model import BikeClassifier, InsufficientData, Prediction
from .strava import StravaGateway, sport_type_of

log = logging.getLogger(__name__)


@dataclass
class Outcome:
    activity_id: int
    status: str
    detail: str = ""
    sport_type: str | None = None
    chosen_gear_id: str | None = None
    chosen_bike: str | None = None
    probabilities: list[dict[str, Any]] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class BikeSelectorService:
    """Holds the gateway and the lazily trained model, and applies the result."""

    def __init__(
        self,
        gateway: StravaGateway | None = None,
        classifier: BikeClassifier | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.gateway = gateway or StravaGateway(self.settings)
        self.classifier = classifier or BikeClassifier(self.settings)
        self._train_lock = threading.Lock()
        self._auto_labelled: set[int] = self._load_auto_labelled()

    # ------------------------------------------------- self-labelled tracking

    def _load_auto_labelled(self) -> set[int]:
        path = self.settings.auto_labelled_path
        if not path.exists():
            return set()
        try:
            return {int(x) for x in json.loads(path.read_text())}
        except (OSError, ValueError, json.JSONDecodeError):  # pragma: no cover - defensive
            return set()

    def _remember_auto_labelled(self, activity_id: int) -> None:
        self._auto_labelled.add(int(activity_id))
        try:
            self.settings.ensure_state_dir()
            recent = sorted(self._auto_labelled)[-2000:]
            self.settings.auto_labelled_path.write_text(json.dumps(recent))
        except OSError as exc:  # pragma: no cover - defensive
            log.warning("Could not persist auto-labelled ids: %s", exc)

    # ---------------------------------------------------------------- model

    def ensure_model(self, *, force: bool = False) -> BikeClassifier:
        """Train (or reload) the model at most once at a time."""
        with self._train_lock:
            if not force and self.classifier.is_trained:
                if self.classifier.age_hours <= self.settings.model_ttl_hours:
                    return self.classifier
            if not force and self.classifier.load():
                return self.classifier

            log.info("Training bike classifier from Strava history…")
            activities = self.gateway.recent_activities()
            report = self.classifier.train(
                activities,
                bike_names=self.gateway.bike_names(),
                exclude_ids=self._auto_labelled,
            )
            log.info("%s", report.summary())
            try:
                self.classifier.save()
            except OSError as exc:  # pragma: no cover - defensive
                log.warning("Could not cache the trained model: %s", exc)
            return self.classifier

    # ----------------------------------------------------------- description

    def build_description(self, existing: str | None, predictions: list[Prediction]) -> str:
        """Append (or refresh) our probability line without touching the user's text."""
        shown = predictions[: max(self.settings.max_probabilities_shown, 1)]
        line = "Bike guess: " + " | ".join(p.format() for p in shown)
        marker = self.settings.description_marker
        if marker:
            line = f"{line} {marker}"

        body = (existing or "").rstrip()
        if marker:
            pattern = re.compile(
                rf"^.*{re.escape(marker)}\s*$", flags=re.MULTILINE
            )
            body = pattern.sub("", body).rstrip()

        return f"{body}\n\n{line}".lstrip() if body else line

    def _has_our_marker(self, description: str | None) -> bool:
        marker = self.settings.description_marker
        return bool(marker) and marker in (description or "")

    # -------------------------------------------------------------- pipeline

    def process_activity(
        self,
        activity_id: int,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> Outcome:
        settings = self.settings
        activity = self.gateway.activity(activity_id)
        sport = sport_type_of(activity)

        if not is_bike_activity(activity):
            return Outcome(activity_id, "skipped", f"{sport} is not a bike activity", sport)
        if settings.skip_trainer and getattr(activity, "trainer", False):
            return Outcome(activity_id, "skipped", "indoor/trainer ride", sport)
        if getattr(activity, "manual", False):
            return Outcome(activity_id, "skipped", "manually entered activity", sport)

        description = getattr(activity, "description", None)
        if not force and self._has_our_marker(description):
            return Outcome(activity_id, "skipped", "already annotated by this app", sport)

        existing_gear = gear_id_of(activity)
        if existing_gear and not settings.overwrite_existing_gear and not force:
            return Outcome(
                activity_id,
                "skipped",
                f"bike already set ({self.classifier.name_for(existing_gear)})",
                sport,
            )

        try:
            classifier = self.ensure_model()
            predictions = classifier.predict(activity)
        except InsufficientData as exc:
            return Outcome(activity_id, "no-model", str(exc), sport)

        best = predictions[0]
        probabilities = [
            {"gear_id": p.gear_id, "name": p.name, "probability": round(p.probability, 4)}
            for p in predictions
        ]

        confident = best.probability >= settings.min_confidence
        gear_to_set = best.gear_id if (settings.set_gear and confident) else None
        if gear_to_set and gear_to_set == existing_gear:
            gear_to_set = None  # nothing to change

        new_description = (
            self.build_description(description, predictions)
            if settings.write_description
            else None
        )
        if new_description is not None and new_description == (description or "").strip():
            new_description = None

        if dry_run:
            return Outcome(
                activity_id,
                "dry-run",
                f"would set {best.name}" if gear_to_set else "would only annotate",
                sport,
                best.gear_id,
                best.name,
                probabilities,
            )

        if gear_to_set is None and new_description is None:
            return Outcome(
                activity_id, "unchanged", "prediction matches what is already there", sport,
                best.gear_id, best.name, probabilities,
            )

        self.gateway.update_activity(
            activity_id, gear_id=gear_to_set, description=new_description
        )
        if gear_to_set:
            self._remember_auto_labelled(activity_id)

        detail = (
            f"set {best.name} ({best.probability:.0%})"
            if gear_to_set
            else f"annotated only ({best.probability:.0%} < {settings.min_confidence:.0%})"
            if not confident
            else "annotated only"
        )
        return Outcome(
            activity_id, "updated", detail, sport, best.gear_id, best.name, probabilities
        )

    def handle_event(self, event: dict[str, Any]) -> Outcome | None:
        """Route a Strava webhook payload. Returns None when it is not for us."""
        if event.get("object_type") != "activity":
            return None
        if event.get("aspect_type") != "create":
            return None
        activity_id = event.get("object_id")
        if activity_id is None:
            return None
        return self.process_activity(int(activity_id))

    def backfill(
        self,
        *,
        days: int = 30,
        limit: int = 50,
        only_missing_gear: bool = True,
        dry_run: bool = False,
    ) -> list[Outcome]:
        """Catch rides a dropped webhook missed."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        outcomes: list[Outcome] = []
        for activity in self.gateway.recent_activities(limit=limit):
            started = getattr(activity, "start_date", None)
            if isinstance(started, datetime):
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                if started < cutoff:
                    break
            if not is_bike_activity(activity):
                continue
            if only_missing_gear and gear_id_of(activity):
                continue
            outcomes.append(
                self.process_activity(int(activity.id), dry_run=dry_run)
            )
        return outcomes
