"""Give a ride a title worth reading instead of Strava's generic default.

Strava auto-names an upload "Morning Ride" ("Morgenausfahrt" on a German-locale
account), "Lunch Ride", "Evening Ride" and so on whenever the rider never
bothers to type one. Those are the only titles we ever touch — see
:func:`is_default_title` — so a name someone actually chose is never clobbered
unless the caller forces it.

The replacement is German: an adjective (how the ride felt, relative to this
bike's own history) plus the bike's category, e.g. "Entspannte Rennrad-Tour"
or "Schnelle Gravel-Tour" — with the ride's longest climbs named after it, if
it had any. The adjective pool is picked deterministically from the activity
id, so re-processing the same ride (a retry, a dry run, a rerun of
`backfill`) always proposes the same title rather than re-rolling the dice.
"""

from __future__ import annotations

import random
import re
from typing import Any

from .features import extract
from .model import Prediction

#: Strava's own auto-generated names, English ("Morning Ride", "Lunch Ride")
#: and German ("Morgenausfahrt", "Mittagsausfahrt"). Only these get replaced;
#: anything a rider typed themselves is left alone.
_DEFAULT_TITLE_RE = re.compile(
    r"^("
    r"(early morning|morning|lunch|afternoon|evening|night)\s+ride"
    r"|(morgen|vormittags|mittags|nachmittags|abend|nacht)s?ausfahrt"
    r")$",
    re.IGNORECASE,
)

#: Cap so a title stays a title, not a paragraph.
MAX_TITLE_LENGTH = 80

#: Fallback bike-category word when none is configured for the gear.
DEFAULT_BIKE_TYPE = "Fahrrad"

# --------------------------------------------------------------- adjectives
#
# Which pool a ride draws from depends on how its average speed compares to
# *this bike's* own training history (see BikeClassifier.speed_baseline) —
# notably slower than usual is "relaxed", notably faster is "brisk", anything
# in between (or a bike with no history yet) is a neutral compliment.

RELAXED: tuple[str, ...] = ("Entspannte", "Gemütliche", "Ruhige", "Lockere", "Beschauliche")
BRISK: tuple[str, ...] = ("Schnelle", "Flotte", "Rasante", "Sportliche", "Zügige")
NEUTRAL: tuple[str, ...] = ("Schöne", "Klassische", "Herrliche", "Feine", "Kleine")

#: How many standard deviations off this bike's average speed counts as
#: "relaxed" or "brisk" rather than just an ordinary ride for it.
_SPEED_Z_THRESHOLD = 0.5
#: Floor on the standard deviation so a bike with almost no spread in its
#: history (or only one ride so far) doesn't get called "brisk" or "relaxed"
#: over a fraction of a km/h.
_MIN_SPEED_STD_KMH = 1.5


def _adjective(
    rng: random.Random, avg_speed_kmh: float, speed_baseline: tuple[float, float] | None
) -> str:
    if speed_baseline is None or avg_speed_kmh != avg_speed_kmh:  # NaN != NaN
        return rng.choice(NEUTRAL)
    mean, std = speed_baseline
    if mean != mean:
        return rng.choice(NEUTRAL)
    spread = max(std, _MIN_SPEED_STD_KMH)
    z = (avg_speed_kmh - mean) / spread
    if z <= -_SPEED_Z_THRESHOLD:
        return rng.choice(RELAXED)
    if z >= _SPEED_Z_THRESHOLD:
        return rng.choice(BRISK)
    return rng.choice(NEUTRAL)


# ------------------------------------------------------------------ climbs


def _climb_segments(activity: Any, *, limit: int, min_grade: float) -> list[str]:
    """Names of the ride's longest climbs, longest first.

    A "climb" is a segment effort whose segment averages at least
    ``min_grade`` percent — Strava's own ``climb_category`` is not used
    because it is often 0 ("uncategorized") for perfectly real climbs.
    Length comes from the effort itself (how much of the segment was
    actually ridden), falling back to the segment's own length.
    """
    efforts = getattr(activity, "segment_efforts", None) or []
    seen: set[str] = set()
    climbs: list[tuple[float, str]] = []
    for effort in efforts:
        segment = getattr(effort, "segment", None)
        grade = getattr(segment, "average_grade", None) if segment is not None else None
        if grade is None or grade < min_grade:
            continue
        name = str(getattr(effort, "name", None) or getattr(segment, "name", None) or "").strip()
        if not name or name in seen:
            continue
        length = getattr(effort, "distance", None)
        if length is None and segment is not None:
            length = getattr(segment, "distance", None)
        if length is None:
            continue
        seen.add(name)
        climbs.append((float(length), name))
    climbs.sort(key=lambda c: c[0], reverse=True)
    return [name for _, name in climbs[:limit]]


def _join_german(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " und " + items[-1]


def is_default_title(name: str | None) -> bool:
    """True for an empty title or one of Strava's own auto-generated defaults."""
    text = (name or "").strip()
    return not text or bool(_DEFAULT_TITLE_RE.match(text))


def generate_title(
    activity: Any,
    prediction: Prediction | None = None,
    *,
    bike_type: str | None = None,
    speed_baseline: tuple[float, float] | None = None,
    climb_limit: int = 2,
    climb_min_grade: float = 3.0,
    seed: int | None = None,
) -> str:
    """Build a title like "Entspannte Rennrad-Tour über Bergstraße".

    ``bike_type`` is the German category word to use (e.g. "Rennrad",
    "Gravel"); it defaults to the predicted bike's own Strava name, or
    :data:`DEFAULT_BIKE_TYPE` if there is no prediction either.
    ``speed_baseline`` is this bike's own (mean, std) average speed in km/h
    from training history, used to pick "relaxed" vs. "brisk" vs. neutral.
    ``seed`` defaults to the activity id so the result is stable across
    reruns; pass an explicit value (or vary it) to get a fresh roll.
    """
    rng = random.Random(seed if seed is not None else getattr(activity, "id", 0))
    resolved_type = bike_type or (prediction.name if prediction else None) or DEFAULT_BIKE_TYPE
    avg_speed_kmh = extract(activity)["avg_speed_kmh"]

    title = f"{_adjective(rng, avg_speed_kmh, speed_baseline)} {resolved_type}-Tour"
    climbs = _climb_segments(activity, limit=climb_limit, min_grade=climb_min_grade)
    if climbs:
        title += f" über {_join_german(climbs)}"
    return title[:MAX_TITLE_LENGTH].rstrip()
