"""Turn a Strava activity into a fixed-length numeric feature vector.

Everything the model needs is present on the *summary* activity returned by
``get_activities``, so training does not need a request per ride.

Missing values are encoded as NaN on purpose — XGBoost learns a default branch
direction for them, which is exactly what we want: "no power meter" is itself a
strong signal about which bike was used.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .strava import BIKE_SPORT_TYPES, sport_type_of

NAN = float("nan")

#: One-hot columns for sport type. Kept explicit so train/predict always align.
SPORT_TYPE_VOCAB: tuple[str, ...] = BIKE_SPORT_TYPES

#: Recording device (head unit) name, hashed into a handful of buckets rather
#: than a per-user vocabulary. Not "which bike" by itself, but riders often
#: pair one head unit with one bike, and it costs nothing extra: it is already
#: present on the summary activity, unlike power-meter laterality (only on the
#: activity's data streams, which would need one extra API call per ride).
DEVICE_HASH_BUCKETS = 4

BASE_FEATURES: tuple[str, ...] = (
    "distance_km",
    "moving_min",
    "elapsed_min",
    "moving_ratio",
    "avg_speed_kmh",
    "max_speed_kmh",
    "speed_burstiness",
    "elev_gain_m",
    "elev_per_km",
    "elev_high_m",
    "elev_low_m",
    "elev_range_m",
    "avg_watts",
    "max_watts",
    "weighted_avg_watts",
    "kilojoules",
    "watts_per_kmh",
    "has_power_meter",
    "has_heartrate",
    "avg_heartrate",
    "max_heartrate",
    "has_cadence",
    "avg_cadence",
    "avg_temp_c",
    "is_trainer",
    "is_commute",
    "is_manual",
    "start_lat",
    "start_lng",
    "start_hour",
    "day_of_week",
    "is_weekend",
    "month",
    "achievement_count",
    "pr_count",
    "athlete_count",
    "has_device_name",
)

FEATURE_NAMES: tuple[str, ...] = (
    BASE_FEATURES
    + tuple(f"sport_{name}" for name in SPORT_TYPE_VOCAB)
    + tuple(f"device_bucket_{i}" for i in range(DEVICE_HASH_BUCKETS))
)


# --------------------------------------------------------------------- coercion


def _num(value: Any) -> float:
    """Coerce a stravalib attribute to a float, or NaN.

    Handles plain numbers, stravalib's float-subclass custom types (``Distance``,
    ``Velocity``), ``timedelta``/``Duration`` (converted to seconds) and pint
    ``Quantity`` objects.
    """
    if value is None:
        return NAN
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, (int, float)):
        return float(value)
    for attr in ("magnitude", "root", "value"):
        inner = getattr(value, attr, None)
        if inner is not None and inner is not value:
            return _num(inner)
    try:
        return float(value)  # pragma: no cover - last resort
    except (TypeError, ValueError):
        return NAN


def _flag(value: Any) -> float:
    """Tri-state boolean: 1.0 true, 0.0 false, NaN unknown."""
    if value is None:
        return NAN
    return 1.0 if bool(value) else 0.0


def _latlng(value: Any) -> tuple[float, float]:
    if value is None:
        return NAN, NAN
    root = getattr(value, "root", value)
    lat = getattr(root, "lat", None)
    lng = getattr(root, "lon", None) or getattr(root, "lng", None)
    if lat is not None and lng is not None:
        return _num(lat), _num(lng)
    try:
        pair = list(root)
    except TypeError:
        return NAN, NAN
    if len(pair) >= 2:
        return _num(pair[0]), _num(pair[1])
    return NAN, NAN


def _safe_div(numerator: float, denominator: float) -> float:
    if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
        return NAN
    return numerator / denominator


def start_datetime(activity: Any) -> datetime | None:
    for attr in ("start_date_local", "start_date"):
        value = getattr(activity, attr, None)
        if isinstance(value, datetime):
            return value
    return None


def gear_id_of(activity: Any) -> str | None:
    value = getattr(activity, "gear_id", None)
    if value is None:
        gear = getattr(activity, "gear", None)
        value = getattr(gear, "id", None) if gear is not None else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def is_bike_activity(activity: Any) -> bool:
    return sport_type_of(activity) in BIKE_SPORT_TYPES


def _device_bucket(device_name: str, buckets: int = DEVICE_HASH_BUCKETS) -> int:
    """Stable hash bucket for a recording-device name.

    Deliberately not Python's built-in ``hash()``: that is salted per-process,
    so a model trained in one process and used to predict in another (exactly
    what happens here — train and predict run in different requests) would see
    a different bucket for the same string and the feature would be noise.
    """
    digest = hashlib.md5(device_name.strip().lower().encode("utf-8")).digest()
    return digest[0] % buckets


# ----------------------------------------------------------------- extraction


def extract(activity: Any) -> dict[str, float]:
    """Build the named feature mapping for a single activity."""
    distance_m = _num(getattr(activity, "distance", None))
    moving_s = _num(getattr(activity, "moving_time", None))
    elapsed_s = _num(getattr(activity, "elapsed_time", None))
    avg_speed_ms = _num(getattr(activity, "average_speed", None))
    max_speed_ms = _num(getattr(activity, "max_speed", None))
    elev_gain = _num(getattr(activity, "total_elevation_gain", None))
    elev_high = _num(getattr(activity, "elev_high", None))
    elev_low = _num(getattr(activity, "elev_low", None))
    avg_watts = _num(getattr(activity, "average_watts", None))
    avg_cadence = _num(getattr(activity, "average_cadence", None))

    distance_km = _safe_div(distance_m, 1000.0)
    avg_speed_kmh = avg_speed_ms * 3.6 if not math.isnan(avg_speed_ms) else NAN
    max_speed_kmh = max_speed_ms * 3.6 if not math.isnan(max_speed_ms) else NAN

    started = start_datetime(activity)
    lat, lng = _latlng(getattr(activity, "start_latlng", None))

    features: dict[str, float] = {
        "distance_km": distance_km,
        "moving_min": _safe_div(moving_s, 60.0),
        "elapsed_min": _safe_div(elapsed_s, 60.0),
        "moving_ratio": _safe_div(moving_s, elapsed_s),
        "avg_speed_kmh": avg_speed_kmh,
        "max_speed_kmh": max_speed_kmh,
        "speed_burstiness": _safe_div(max_speed_kmh, avg_speed_kmh),
        "elev_gain_m": elev_gain,
        "elev_per_km": _safe_div(elev_gain, distance_km),
        "elev_high_m": elev_high,
        "elev_low_m": elev_low,
        "elev_range_m": (
            elev_high - elev_low
            if not (math.isnan(elev_high) or math.isnan(elev_low))
            else NAN
        ),
        "avg_watts": avg_watts,
        "max_watts": _num(getattr(activity, "max_watts", None)),
        "weighted_avg_watts": _num(getattr(activity, "weighted_average_watts", None)),
        "kilojoules": _num(getattr(activity, "kilojoules", None)),
        # Rolling resistance proxy: watts needed to hold a given speed separates
        # a road bike from a loaded commuter or an MTB rather well.
        "watts_per_kmh": _safe_div(avg_watts, avg_speed_kmh),
        "has_power_meter": _flag(getattr(activity, "device_watts", None)),
        "has_heartrate": _flag(getattr(activity, "has_heartrate", None)),
        "avg_heartrate": _num(getattr(activity, "average_heartrate", None)),
        "max_heartrate": _num(getattr(activity, "max_heartrate", None)),
        "has_cadence": 0.0 if math.isnan(avg_cadence) else 1.0,
        "avg_cadence": avg_cadence,
        "avg_temp_c": _num(getattr(activity, "average_temp", None)),
        "is_trainer": _flag(getattr(activity, "trainer", None)),
        "is_commute": _flag(getattr(activity, "commute", None)),
        "is_manual": _flag(getattr(activity, "manual", None)),
        "start_lat": lat,
        "start_lng": lng,
        "start_hour": float(started.hour + started.minute / 60) if started else NAN,
        "day_of_week": float(started.weekday()) if started else NAN,
        "is_weekend": float(started.weekday() >= 5) if started else NAN,
        "month": float(started.month) if started else NAN,
        "achievement_count": _num(getattr(activity, "achievement_count", None)),
        "pr_count": _num(getattr(activity, "pr_count", None)),
        "athlete_count": _num(getattr(activity, "athlete_count", None)),
    }

    sport = sport_type_of(activity)
    for name in SPORT_TYPE_VOCAB:
        features[f"sport_{name}"] = 1.0 if sport == name else 0.0

    device_name = getattr(activity, "device_name", None)
    device_name = str(device_name).strip() if device_name else ""
    features["has_device_name"] = 1.0 if device_name else 0.0
    for i in range(DEVICE_HASH_BUCKETS):
        features[f"device_bucket_{i}"] = 0.0
    if device_name:
        features[f"device_bucket_{_device_bucket(device_name)}"] = 1.0

    return features


def to_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([features.get(name, NAN) for name in FEATURE_NAMES], dtype=np.float32)


def to_matrix(activities: list[Any]) -> np.ndarray:
    if not activities:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([to_vector(extract(activity)) for activity in activities])
