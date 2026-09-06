from __future__ import annotations

import pytest

from bike_selector.model import Prediction
from bike_selector.titles import generate_title, is_default_title

from .conftest import ROAD, FakeActivity, FakeSegment, FakeSegmentEffort

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


def test_title_has_the_expected_shape():
    activity = FakeActivity(id=1234)
    title = generate_title(activity, PREDICTION, bike_type="Rennrad")
    assert title.endswith("Rennrad-Tour")
    assert title.split()[0][0].isupper()


def test_same_activity_id_is_deterministic():
    activity = FakeActivity(id=4242)
    first = generate_title(activity, PREDICTION)
    second = generate_title(activity, PREDICTION)
    assert first == second


def test_different_activities_can_yield_different_titles():
    titles = {generate_title(FakeActivity(id=i), PREDICTION) for i in range(20)}
    assert len(titles) > 1


def test_missing_prediction_falls_back_to_default_bike_type():
    activity = FakeActivity(id=55)
    title = generate_title(activity, None)
    assert title
    assert "Fahrrad-Tour" in title


def test_bike_type_overrides_predicted_bike_name():
    activity = FakeActivity(id=9)
    title = generate_title(activity, PREDICTION, bike_type="Gravel")
    assert "Gravel-Tour" in title
    assert ROAD.name not in title


def test_no_bike_type_falls_back_to_predicted_bike_name():
    activity = FakeActivity(id=9)
    title = generate_title(activity, PREDICTION)
    assert f"{ROAD.name}-Tour" in title


# -------------------------------------------------------------------- speed


def test_relaxed_ride_gets_a_relaxed_adjective():
    activity = FakeActivity(id=1, average_speed=5.0)  # 18 km/h
    title = generate_title(activity, PREDICTION, speed_baseline=(30.0, 2.0))
    assert title.split()[0] in ("Entspannte", "Gemütliche", "Ruhige", "Lockere", "Beschauliche")


def test_brisk_ride_gets_a_brisk_adjective():
    activity = FakeActivity(id=1, average_speed=12.0)  # 43.2 km/h
    title = generate_title(activity, PREDICTION, speed_baseline=(25.0, 2.0))
    assert title.split()[0] in ("Schnelle", "Flotte", "Rasante", "Sportliche", "Zügige")


def test_no_baseline_still_produces_a_title():
    activity = FakeActivity(id=1)
    assert generate_title(activity, PREDICTION, speed_baseline=None)


# ------------------------------------------------------------------- climbs


def test_climb_segments_are_named_in_title():
    activity = FakeActivity(
        id=2,
        segment_efforts=[
            FakeSegmentEffort(
                name="Kurzer Anstieg",
                distance=500.0,
                segment=FakeSegment(name="Kurzer Anstieg", distance=500.0, average_grade=6.0),
            ),
            FakeSegmentEffort(
                name="Langer Berg",
                distance=3000.0,
                segment=FakeSegment(name="Langer Berg", distance=3000.0, average_grade=8.0),
            ),
        ],
    )
    title = generate_title(activity, PREDICTION, bike_type="Rennrad")
    assert "Langer Berg" in title
    assert "Kurzer Anstieg" in title
    # Longest climb first.
    assert title.index("Langer Berg") < title.index("Kurzer Anstieg")


def test_flat_segments_are_not_treated_as_climbs():
    activity = FakeActivity(
        id=3,
        segment_efforts=[
            FakeSegmentEffort(
                name="Flache Gerade",
                distance=5000.0,
                segment=FakeSegment(name="Flache Gerade", distance=5000.0, average_grade=1.0),
            ),
        ],
    )
    title = generate_title(activity, PREDICTION, bike_type="Rennrad")
    assert "Flache Gerade" not in title
    assert title.endswith("Rennrad-Tour")


def test_climb_count_is_limited():
    efforts = [
        FakeSegmentEffort(
            name=f"Berg {i}",
            distance=float(1000 + i),
            segment=FakeSegment(name=f"Berg {i}", distance=float(1000 + i), average_grade=5.0),
        )
        for i in range(5)
    ]
    activity = FakeActivity(id=4, segment_efforts=efforts)
    title = generate_title(activity, PREDICTION, bike_type="Rennrad", climb_limit=1)
    assert title.count("Berg") == 1


def test_no_segments_produces_a_plain_tour_title():
    activity = FakeActivity(id=5, segment_efforts=[])
    title = generate_title(activity, PREDICTION, bike_type="Rennrad")
    assert title.endswith("Rennrad-Tour")
    assert "über" not in title
