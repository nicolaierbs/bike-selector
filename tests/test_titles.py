from __future__ import annotations

import pytest

from bike_selector.model import Prediction
from bike_selector.titles import (
    TITLE_STYLES,
    generate_title,
    is_default_title,
)

from .conftest import ROAD, FakeActivity

PREDICTION = Prediction(ROAD.id, ROAD.name, 0.87)


# ------------------------------------------------------------- default detection


@pytest.mark.parametrize(
    "name",
    [
        None,
        "",
        "  ",
        "Morning Ride",
        "morning ride",
        "Lunch Ride",
        "Evening Ride",
        "Night Ride",
        "Morgenausfahrt",
        "morgenausfahrt",
        "Mittagsausfahrt",
        "Nachmittagsausfahrt",
        "Abendausfahrt",
        "Nachtausfahrt",
    ],
)
def test_default_titles_are_detected(name):
    assert is_default_title(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "Sunday loop with the club",
        "Morning Run",
        "Ride to Grandma's",
        "MORNING RIDESHARE",
        "Sonntagsrunde mit dem Verein",
        "Ausfahrt zur Oma",
    ],
)
def test_custom_titles_are_not_treated_as_default(name):
    assert is_default_title(name) is False


# ---------------------------------------------------------------- generation


def test_generated_title_is_nonempty_and_bounded():
    activity = FakeActivity(id=1234)
    title = generate_title(activity, PREDICTION)
    assert title
    assert len(title) <= 80


def test_same_activity_id_is_deterministic():
    activity = FakeActivity(id=4242)
    first = generate_title(activity, PREDICTION)
    second = generate_title(activity, PREDICTION)
    assert first == second


def test_different_activities_can_yield_different_titles():
    titles = {generate_title(FakeActivity(id=i), PREDICTION) for i in range(20)}
    assert len(titles) > 1


@pytest.mark.parametrize("style", TITLE_STYLES)
def test_each_style_produces_a_title(style):
    activity = FakeActivity(id=99)
    title = generate_title(activity, PREDICTION, style=style)
    assert title


def test_unknown_style_falls_back_to_any():
    # A bogus style should still produce something sensible rather than raising.
    activity = FakeActivity(id=7)
    assert generate_title(activity, PREDICTION, style="not-a-style")


def test_placeholders_are_filled_in_not_left_as_braces():
    for i in range(30):
        title = generate_title(FakeActivity(id=i), PREDICTION)
        assert "{" not in title and "}" not in title


def test_missing_prediction_still_produces_a_title():
    activity = FakeActivity(id=55)
    assert generate_title(activity, None)
