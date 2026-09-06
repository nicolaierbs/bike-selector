"""Multiclass XGBoost classifier over past rides: features -> which bike.

Depends on numpy + xgboost + scikit-learn. scikit-learn is not used directly, but
``XGBClassifier`` is xgboost's sklearn-API wrapper and imports it at call time.
pandas is deliberately avoided — memory is tight on a 512 MB free tier — and the
label encoding and stratified cross-validation are implemented here rather than
pulled from ``sklearn.model_selection`` to keep the hot path free of extra
imports.
"""

from __future__ import annotations

import logging
import math
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBClassifier

from .config import Settings, get_settings
from .features import (
    BASE_FEATURES,
    FEATURE_NAMES,
    extract,
    gear_id_of,
    is_bike_activity,
    start_datetime,
    to_vector,
)

log = logging.getLogger(__name__)

MODEL_FORMAT_VERSION = 2


class InsufficientData(RuntimeError):
    """Not enough labelled history to train a usable classifier."""


@dataclass(frozen=True)
class Prediction:
    gear_id: str
    name: str
    probability: float

    def format(self, digits: int = 0) -> str:
        return f"{self.name} {self.probability * 100:.{digits}f}%"


@dataclass
class TrainingReport:
    n_samples: int
    n_classes: int
    class_counts: dict[str, int] = field(default_factory=dict)
    cv_accuracy: float | None = None
    cv_folds: int = 0
    baseline_accuracy: float | None = None
    top_features: list[tuple[str, float]] = field(default_factory=list)
    trained_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dropped_bikes: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Trained on {self.n_samples} rides across {self.n_classes} bikes "
            f"({self.trained_at:%Y-%m-%d %H:%M UTC})"
        ]
        if self.cv_accuracy is not None:
            baseline = (
                f" (always-guess-most-common baseline: {self.baseline_accuracy:.1%})"
                if self.baseline_accuracy is not None
                else ""
            )
            lines.append(
                f"{self.cv_folds}-fold CV accuracy: {self.cv_accuracy:.1%}{baseline}"
            )
        else:
            lines.append("Cross-validation skipped (a bike has fewer than 2 labelled rides)")
        for name, count in sorted(self.class_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:<32} {count:>5} rides")
        for name, count in sorted(self.dropped_bikes.items(), key=lambda kv: -kv[1]):
            lines.append(f"  (dropped) {name:<22} {count:>5} rides — below MIN_RIDES_PER_BIKE")
        if self.top_features:
            lines.append("Most informative features:")
            for name, gain in self.top_features:
                lines.append(f"  {name:<32} {gain:>8.3f}")
        return "\n".join(lines)


def _build_estimator(n_classes: int, n_samples: int) -> XGBClassifier:
    """Conservative hyperparameters: personal ride histories are small datasets.

    ``objective``/``num_class`` are deliberately left unset. XGBClassifier derives
    them from the labels, and hard-coding ``multi:softprob`` breaks the two-bike
    case, where the wrapper switches to a binary objective and never populates
    ``num_class``. ``predict_proba`` returns one column per class either way.
    """
    n_estimators = int(min(400, max(60, n_samples // 2)))
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_lambda=1.5,
        gamma=0.0,
        tree_method="hist",
        n_jobs=1,  # single worker; the free tier has one shared vCPU
        verbosity=0,
    )


def _as_aware(stamp: datetime) -> datetime:
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)


def _causal_prev_gear(
    stamps: list[datetime | None], labels: list[str]
) -> tuple[list[str | None], list[float], str | None, datetime | None]:
    """For each row, the gear and day-gap of the closest strictly-earlier
    labelled ride in this history — "what bike were they on right before this
    one", a strong real-world signal for bikes that otherwise ride alike.

    Rows without a timestamp get no context and never advance the running
    state, since their true place in the sequence is unknown. Only genuinely
    earlier rides can inform a row — this is a causal, leakage-free feature:
    a row's own label never affects its own context.

    Also returns the final (most recent) gear and timestamp, which becomes
    the context for a brand-new, not-yet-labelled ride at predict time.
    """
    order = sorted(
        (i for i, s in enumerate(stamps) if s is not None),
        key=lambda i: _as_aware(stamps[i]),
    )
    prev_gear: list[str | None] = [None] * len(stamps)
    prev_days: list[float] = [math.nan] * len(stamps)
    last_gear: str | None = None
    last_time: datetime | None = None
    for i in order:
        stamp = _as_aware(stamps[i])
        if last_time is not None:
            prev_gear[i] = last_gear
            prev_days[i] = max((stamp - last_time).total_seconds() / 86400, 0.0)
        last_gear, last_time = labels[i], stamp
    return prev_gear, prev_days, last_gear, last_time


def _stratified_folds(y: np.ndarray, n_folds: int, seed: int = 0) -> list[np.ndarray]:
    """Indices per fold, keeping the class mix roughly constant in each fold."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for position, sample in enumerate(idx):
            folds[position % n_folds].append(int(sample))
    return [np.array(sorted(fold), dtype=int) for fold in folds]


class BikeClassifier:
    """Trains on labelled history and predicts a probability per bike."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.estimator: XGBClassifier | None = None
        self.labels: list[str] = []
        self.bike_names: dict[str, str] = {}
        self.report: TrainingReport | None = None
        self.feature_names: tuple[str, ...] = FEATURE_NAMES
        # Context for the "what bike were they on right before this one" feature.
        self.last_gear_id: str | None = None
        self.last_ride_at: datetime | None = None
        #: Per-bike (mean, std) average speed in km/h, from training history —
        #: used to tell a "relaxed" ride from a "brisk" one for that bike.
        self.speed_baselines: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------- properties

    @property
    def is_trained(self) -> bool:
        return self.estimator is not None and len(self.labels) >= 2

    @property
    def age_hours(self) -> float:
        if self.report is None:
            return math.inf
        delta = datetime.now(timezone.utc) - self.report.trained_at
        return delta.total_seconds() / 3600

    def name_for(self, gear_id: str) -> str:
        return self.bike_names.get(gear_id, gear_id)

    def speed_baseline(self, gear_id: str) -> tuple[float, float] | None:
        """This bike's own (mean, std) average speed in km/h, if it has history."""
        return self.speed_baselines.get(gear_id)

    # ---------------------------------------------------------------- dataset

    def build_dataset(
        self,
        activities: list[Any],
        *,
        exclude_ids: set[int] | None = None,
    ) -> tuple[np.ndarray, list[str], list[datetime | None]]:
        """Feature matrix, gear labels and start times for labelled bike rides."""
        exclude_ids = exclude_ids or set()
        rows: list[np.ndarray] = []
        labels: list[str] = []
        stamps: list[datetime | None] = []

        for activity in activities:
            if not is_bike_activity(activity):
                continue
            gear_id = gear_id_of(activity)
            if not gear_id:
                continue
            activity_id = getattr(activity, "id", None)
            if activity_id is not None and int(activity_id) in exclude_ids:
                # Labelled by this app, not by a human — training on it would
                # just feed the model its own guesses back.
                continue
            if getattr(activity, "manual", False):
                continue
            rows.append(to_vector(extract(activity)))
            labels.append(gear_id)
            stamps.append(start_datetime(activity))

        matrix = np.vstack(rows) if rows else np.empty((0, len(FEATURE_NAMES)), np.float32)
        return matrix, labels, stamps

    def _sample_weights(self, stamps: list[datetime | None], y: np.ndarray) -> np.ndarray:
        """Recency decay times inverse-frequency, so rare bikes stay learnable."""
        half_life = max(self.settings.recency_half_life_days, 1.0)
        now = datetime.now(timezone.utc)
        recency = np.ones(len(stamps), dtype=np.float32)
        for i, stamp in enumerate(stamps):
            if stamp is None:
                continue
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            age_days = max((now - stamp).total_seconds() / 86400, 0.0)
            recency[i] = float(0.5 ** (age_days / half_life))

        counts = np.bincount(y, minlength=int(y.max()) + 1).astype(np.float32)
        balance = (len(y) / (len(counts) * np.maximum(counts, 1)))[y]

        weights = recency * balance
        mean = float(weights.mean())
        return weights / mean if mean > 0 else np.ones_like(weights)

    # ------------------------------------------------------- prev-gear context

    def _prev_gear_feature_names(self, labels: list[str] | None = None) -> tuple[str, ...]:
        labels = self.labels if labels is None else labels
        return (
            "has_prev_ride",
            "days_since_prev_ride",
            *(f"prev_gear_{gear}" for gear in labels),
            "prev_gear_other",
        )

    def _augment_prev_gear(
        self,
        base: np.ndarray,
        prev_gear_ids: list[str | None],
        prev_days: list[float],
    ) -> np.ndarray:
        """Append the "previous gear" one-hot block (sized to ``self.labels``)
        onto an already-built base feature matrix."""
        index = {gear: i for i, gear in enumerate(self.labels)}
        extra = np.zeros((len(prev_gear_ids), 2 + len(self.labels) + 1), dtype=np.float32)
        for row, (gear, days) in enumerate(zip(prev_gear_ids, prev_days, strict=True)):
            if gear is None:
                extra[row, 1] = math.nan  # days_since_prev_ride: unknown
                continue
            extra[row, 0] = 1.0  # has_prev_ride
            extra[row, 1] = days
            column = index.get(gear)
            extra[row, 2 + column if column is not None else 2 + len(self.labels)] = 1.0
        return np.hstack([base, extra])

    def _speed_baselines(
        self, matrix: np.ndarray, labels: list[str]
    ) -> dict[str, tuple[float, float]]:
        """Per-bike (mean, std) avg speed in km/h, from that bike's own rows.

        Must run on the base feature matrix, before ``_augment_prev_gear``
        appends columns after it.
        """
        idx = BASE_FEATURES.index("avg_speed_kmh")
        column = matrix[:, idx]
        stats: dict[str, tuple[float, float]] = {}
        for gear in set(labels):
            speeds = column[[label == gear for label in labels]]
            speeds = speeds[~np.isnan(speeds)]
            if speeds.size == 0:
                continue
            stats[gear] = (float(speeds.mean()), float(speeds.std()))
        return stats

    def _days_since_last_ride(self, activity: Any) -> float:
        if self.last_ride_at is None:
            return math.nan
        started = start_datetime(activity)
        if started is None:
            return math.nan
        return max((_as_aware(started) - _as_aware(self.last_ride_at)).total_seconds() / 86400, 0.0)

    # --------------------------------------------------------------- training

    def train(
        self,
        activities: list[Any],
        *,
        bike_names: dict[str, str] | None = None,
        exclude_ids: set[int] | None = None,
    ) -> TrainingReport:
        started = time.perf_counter()
        matrix, raw_labels, stamps = self.build_dataset(activities, exclude_ids=exclude_ids)
        if not raw_labels:
            raise InsufficientData(
                "No rides with a bike attached. Assign bikes to a few past rides in Strava first."
            )

        # Computed over the *raw* (pre-rarity-filter) history so a swap to/from a
        # since-dropped rare bike is still visible context, before self.labels
        # narrows things down to the "other" bucket.
        prev_gear_ids, prev_days, self.last_gear_id, self.last_ride_at = _causal_prev_gear(
            stamps, raw_labels
        )

        names = dict(bike_names or {})
        counts: dict[str, int] = {}
        for label in raw_labels:
            counts[label] = counts.get(label, 0) + 1

        min_rides = max(self.settings.min_rides_per_bike, 1)
        keep = {gear for gear, count in counts.items() if count >= min_rides}
        dropped = {
            names.get(gear, gear): count for gear, count in counts.items() if gear not in keep
        }
        if len(keep) < 2:
            # Relax rather than fail outright: two bikes with a handful of rides
            # each still beats no prediction.
            keep = {gear for gear, count in counts.items() if count >= 2}
            dropped = {
                names.get(gear, gear): count for gear, count in counts.items() if gear not in keep
            }
        if len(keep) < 2:
            raise InsufficientData(
                f"Only {len(counts)} bike(s) with enough labelled rides "
                f"({counts}). Need at least 2 to make a prediction."
            )

        mask = np.array([label in keep for label in raw_labels], dtype=bool)
        matrix = matrix[mask]
        kept_labels = [label for label, ok in zip(raw_labels, mask, strict=True) if ok]
        kept_stamps = [stamp for stamp, ok in zip(stamps, mask, strict=True) if ok]
        kept_prev_gear = [g for g, ok in zip(prev_gear_ids, mask, strict=True) if ok]
        kept_prev_days = [d for d, ok in zip(prev_days, mask, strict=True) if ok]

        self.labels = sorted(keep)
        index = {gear: i for i, gear in enumerate(self.labels)}
        y = np.array([index[label] for label in kept_labels], dtype=np.int32)
        weights = self._sample_weights(kept_stamps, y)
        self.speed_baselines = self._speed_baselines(matrix, kept_labels)
        matrix = self._augment_prev_gear(matrix, kept_prev_gear, kept_prev_days)
        self.feature_names = FEATURE_NAMES + self._prev_gear_feature_names()

        cv_accuracy, cv_folds, baseline = self._cross_validate(matrix, y, weights)

        self.estimator = _build_estimator(len(self.labels), len(y))
        self.estimator.fit(matrix, y, sample_weight=weights, verbose=False)
        self.bike_names = names

        self.report = TrainingReport(
            n_samples=int(len(y)),
            n_classes=len(self.labels),
            class_counts={names.get(g, g): counts[g] for g in self.labels},
            cv_accuracy=cv_accuracy,
            cv_folds=cv_folds,
            baseline_accuracy=baseline,
            top_features=self._top_features(),
            dropped_bikes=dropped,
        )
        log.info(
            "Trained in %.1fs — %s", time.perf_counter() - started, self.report.summary().splitlines()[0]
        )
        return self.report

    def _cross_validate(
        self, matrix: np.ndarray, y: np.ndarray, weights: np.ndarray
    ) -> tuple[float | None, int, float | None]:
        counts = np.bincount(y)
        smallest = int(counts.min())
        if smallest < 2 or len(y) < 8:
            return None, 0, None

        n_folds = int(min(5, smallest))
        folds = _stratified_folds(y, n_folds)
        correct = 0
        total = 0
        for fold in folds:
            if fold.size == 0:
                continue
            train_idx = np.setdiff1d(np.arange(len(y)), fold)
            if len(np.unique(y[train_idx])) < len(counts):
                continue  # a class vanished from this training split
            estimator = _build_estimator(len(counts), len(train_idx))
            estimator.fit(matrix[train_idx], y[train_idx], sample_weight=weights[train_idx])
            predicted = estimator.predict(matrix[fold])
            correct += int((predicted == y[fold]).sum())
            total += int(fold.size)

        if total == 0:
            return None, 0, None
        baseline = float(counts.max() / counts.sum())
        return correct / total, n_folds, baseline

    def _top_features(self, k: int = 8) -> list[tuple[str, float]]:
        if self.estimator is None:
            return []
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None:
            return []
        names = self.feature_names or FEATURE_NAMES
        order = np.argsort(importances)[::-1][:k]
        return [
            (names[i], float(importances[i]))
            for i in order
            if i < len(names) and importances[i] > 0
        ]

    # ------------------------------------------------------------- prediction

    def predict(self, activity: Any) -> list[Prediction]:
        """Probability per bike, highest first."""
        if not self.is_trained or self.estimator is None:
            raise InsufficientData("Model is not trained yet.")
        base = to_vector(extract(activity)).reshape(1, -1)
        vector = self._augment_prev_gear(
            base, [self.last_gear_id], [self._days_since_last_ride(activity)]
        )
        probabilities = self.estimator.predict_proba(vector)[0]
        predictions = [
            Prediction(gear_id=gear, name=self.name_for(gear), probability=float(p))
            for gear, p in zip(self.labels, probabilities, strict=True)
        ]
        return sorted(predictions, key=lambda p: p.probability, reverse=True)

    # ------------------------------------------------------------ persistence

    def save(self, path: Path | None = None) -> Path:
        path = path or self.settings.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": MODEL_FORMAT_VERSION,
            "estimator": self.estimator,
            "labels": self.labels,
            "bike_names": self.bike_names,
            "report": self.report,
            "feature_names": list(self.feature_names),
            "last_gear_id": self.last_gear_id,
            "last_ride_at": self.last_ride_at,
            "speed_baselines": self.speed_baselines,
        }
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        return path

    def load(self, path: Path | None = None) -> bool:
        """Restore a cached model. Returns False when it is missing or stale."""
        path = path or self.settings.model_path
        if not path.exists():
            return False
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)  # noqa: S301 - our own file, our own process
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Could not read cached model %s: %s", path, exc)
            return False

        if payload.get("version") != MODEL_FORMAT_VERSION:
            log.info("Cached model has an old format version; retraining.")
            return False

        labels = list(payload.get("labels", []))
        stored_feature_names = tuple(payload.get("feature_names", ()))
        expected_feature_names = FEATURE_NAMES + self._prev_gear_feature_names(labels)
        if stored_feature_names != expected_feature_names:
            log.info("Feature set changed since the model was cached; retraining.")
            return False

        self.estimator = payload["estimator"]
        self.labels = labels
        self.bike_names = dict(payload.get("bike_names") or {})
        self.report = payload.get("report")
        self.feature_names = stored_feature_names
        self.last_gear_id = payload.get("last_gear_id")
        self.last_ride_at = payload.get("last_ride_at")
        self.speed_baselines = dict(payload.get("speed_baselines") or {})

        if self.age_hours > self.settings.model_ttl_hours:
            log.info("Cached model is %.1f h old; retraining.", self.age_hours)
            return False
        return self.is_trained
