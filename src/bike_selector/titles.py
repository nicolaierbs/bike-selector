"""Give a ride a title worth reading instead of Strava's generic default.

Strava auto-names an upload "Morning Ride" ("Morgenausfahrt" on a German-locale
account), "Lunch Ride", "Evening Ride" and so on whenever the rider never
bothers to type one. Those are the only titles we ever touch — see
:func:`is_default_title` — so a name someone actually chose is never clobbered
unless the caller forces it.

The replacement is German, picked from five flavours (epic / funny /
historical / random / puns) and rendered against a handful of ride stats.
Selection is seeded by the activity id, so re-processing the same ride (a
retry, a dry run, a rerun of `backfill`) always proposes the same title rather
than re-rolling the dice.
"""

from __future__ import annotations

import random
import re
from typing import Any

from .features import _num, start_datetime
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

TITLE_STYLES: tuple[str, ...] = ("epic", "funny", "historical", "random", "puns")

# --------------------------------------------------------------- templates
#
# All in German — the app writes back to a German Strava account.

EPIC: tuple[str, ...] = (
    "Die {distance} km Odyssee",
    "Eroberung des {elevation} m Gipfels",
    "Legende des {bike} am {weekday}",
    "Aufstieg des eisernen {bike}",
    "Der {month}-Kreuzzug",
    "Sage der tausend Watt",
    "Chroniken des {hour_word}-Kriegers",
    "Die große {distance} km Expedition",
    "Imperium des Asphalts",
    "Das unaufhaltsame {bike}",
    "Herrschaft des {weekday}-Pelotons",
    "Himmelfahrt: {elevation} Höhenmeter näher an den Göttern",
)

FUNNY: tuple[str, ...] = (
    "Rettet meine Beine bei km {distance}",
    "Ich bereue alles ({distance} km Geschichte)",
    "Der Snack war die eigentliche Leistung",
    "{weekday}-Leiden, präsentiert vom {bike}",
    "{bike} gegen die Schwerkraft: Runde 2",
    "Nur ein kleiner {distance} km Umweg vom Sofa",
    "Angetrieben von Kaffee und schlechten Entscheidungen",
    "Mein Sattel hat Beschwerde eingelegt",
    "Auf der Jagd nach Strava-Segmenten, gefangen: keins",
    "Die {hour_word}-Ausfahrt, die niemand wollte",
    "{elevation} Höhenmeter der Sinnfrage",
    "Definitiv nicht zu spät zum {hour_word}-Kaffee",
)

HISTORICAL: tuple[str, ...] = (
    "Hannibals {elevation} m Alpenüberquerung",
    "Paul Reveres {hour_word}-Ritt, nachgestellt",
    "Die Attacke der {bike}-Brigade",
    "Cäsars {distance} km Rubikon",
    "Das Trojanische {bike}",
    "Marco Polos {distance} km Seidenstraßen-Umweg",
    "Der {weekday}-Tee von Boston",
    "Magellans Weltumrundung (Lokale Ausgabe, {distance} km)",
    "Napoleons Rückzug aus dem {month}",
    "Der Plan B der Gebrüder Wright",
    "Spartacus führt die {weekday}-Ausreißergruppe an",
    "Das {bike}, Excalibur des {month}",
)

RANDOM: tuple[str, ...] = (
    "Bermuda-Dreieck, aber als Radweg",
    "Gummiente auf Erkundungstour",
    "Ausfahrt des Jahrhunderts (vermutlich)",
    "Schrödingers Sprint",
    "Das {bike} betritt ein Wurmloch",
    "{distance} km bis zum Sinn des Lebens",
    "Verfolgungsjagd auf ein Eichhörnchen",
    "Die große {month}-Pizza-Jagd",
    "Steppenläufer-Sprint",
    "Ein wildes {bike} erscheint",
    "Irgendwo zwischen {weekday} und Narnia",
    "Koordinaten unbekannt, Stimmung einwandfrei",
)

PUNS: tuple[str, ...] = (
    "Volle Kette voraus",
    "Kettenreaktion pur",
    "Sattelfest nach {distance} km",
    "Alles im grünen Radl-Bereich",
    "Speichenkalypse Now",
    "Radikal unterwegs mit dem {bike}",
    "Nabenschau am {weekday}",
    "Der Lenker lügt nie",
    "Reifen, Rost und Rekorde",
    "Kette rechts, Ausrede links",
    "Sattelschlepper im {month}-Einsatz",
    "Zwei Räder, ein {bike}, kein Plan",
    "Radler-Ehre auf {elevation} Höhenmetern",
    "Pedal zum Metall, {hour_word}-Ausgabe",
    "Ganz schön abgefahren: {distance} km",
)

_POOLS: dict[str, tuple[str, ...]] = {
    "epic": EPIC,
    "funny": FUNNY,
    "historical": HISTORICAL,
    "random": RANDOM,
    "puns": PUNS,
}

_HOUR_WORDS: tuple[tuple[int, str], ...] = (
    (6, "Nacht"),
    (11, "Morgen"),
    (14, "Mittag"),
    (18, "Nachmittag"),
    (22, "Abend"),
    (24, "Nacht"),
)

_WEEKDAYS: tuple[str, ...] = (
    "Montag",
    "Dienstag",
    "Mittwoch",
    "Donnerstag",
    "Freitag",
    "Samstag",
    "Sonntag",
)

_MONTHS: tuple[str, ...] = (
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
)


def _hour_word(hour: int) -> str:
    for ceiling, word in _HOUR_WORDS:
        if hour < ceiling:
            return word
    return "Nacht"  # pragma: no cover - unreachable, hour is always < 24


def is_default_title(name: str | None) -> bool:
    """True for an empty title or one of Strava's own auto-generated defaults."""
    text = (name or "").strip()
    return not text or bool(_DEFAULT_TITLE_RE.match(text))


def _context(activity: Any, prediction: Prediction | None) -> dict[str, str]:
    started = start_datetime(activity)
    distance_km = _num(getattr(activity, "distance", None)) / 1000.0
    elevation_m = _num(getattr(activity, "total_elevation_gain", None))
    return {
        "bike": prediction.name if prediction else "Fahrrad",
        "distance": f"{distance_km:.0f}" if distance_km == distance_km else "??",  # NaN != NaN
        "elevation": f"{elevation_m:.0f}" if elevation_m == elevation_m else "??",
        "weekday": _WEEKDAYS[started.weekday()] if started else "Irgendwann",
        "month": _MONTHS[started.month - 1] if started else "Irgendwann",
        "hour_word": _hour_word(started.hour) if started else "Tag",
    }


def generate_title(
    activity: Any,
    prediction: Prediction | None = None,
    *,
    style: str = "any",
    seed: int | None = None,
) -> str:
    """Pick and render one title template.

    ``style`` is one of :data:`TITLE_STYLES`, or ``"any"`` to let the RNG pick a
    flavour too. ``seed`` defaults to the activity id so the result is stable
    across reruns; pass an explicit value (or vary it) to get a fresh roll.
    """
    rng = random.Random(seed if seed is not None else getattr(activity, "id", 0))
    pool_name = style if style in _POOLS else rng.choice(TITLE_STYLES)
    template = rng.choice(_POOLS[pool_name])
    title = template.format(**_context(activity, prediction))
    return title[:MAX_TITLE_LENGTH].rstrip()
