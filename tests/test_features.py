from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pytest

from bike_selector.features import (
    FEATURE_NAMES,
    _num,
    extract,
    gear_id_of,
    is_bike_activity,
    to_matrix,
    to_vector,
)

from .conftest import FakeActivity


def test_vector_length_matches_feature_names():
    vector = to_vector(extract(FakeActivity(id=1)))
    assert vector.shape == (len(FEATURE_NAMES),)
    assert vector.dtype == np.float32


def test_units_are_converted():
    features = extract(FakeActivity(id=1, distance=30_000, average_speed=8.0))
    assert features["distance_km"] == pytest.approx(30.0)
    assert features["avg_speed_kmh"] == pytest.approx(28.8)
    assert features["moving_min"] == pytest.approx(60.0)


def test_derived_ratios():
    features = extract(
        FakeActivity(
            id=1,
            distance=50_000,
            total_elevation_gain=1000,
            moving_time=timedelta(minutes=100),
            elapsed_time=timedelta(minutes=125),
        )
    )
    assert features["elev_per_km"] == pytest.approx(20.0)
    assert features["moving_ratio"] == pytest.approx(0.8)


def test_missing_values_become_nan_not_zero():
    """NaN matters: XGBoost learns a branch for it, 0.0 would be a real reading."""
    features = extract(
        FakeActivity(id=1, average_watts=None, average_cadence=None, average_heartrate=None)
    )
    assert math.isnan(features["avg_watts"])
    assert math.isnan(features["avg_cadence"])
    assert features["has_cadence"] == 0.0


def test_sport_type_one_hot_is_exclusive():
    features = extract(FakeActivity(id=1, sport_type="Ride"))
    assert features["sport_Ride"] == 1.0
    assert sum(v for k, v in features.items() if k.startswith("sport_")) == 1.0
    assert is_bike_activity(FakeActivity(id=1, sport_type="Ride"))


def test_unknown_sport_type_is_all_zeroes_and_not_a_bike():
    activity = FakeActivity(id=1, sport_type="Run")
    features = extract(activity)
    assert sum(v for k, v in features.items() if k.startswith("sport_")) == 0.0
    assert not is_bike_activity(activity)


def test_virtual_and_other_ride_subtypes_are_not_bike_activities():
    """Only sport_type == "Ride" is eligible: virtual rides always use the same
    bike, so they carry no signal; other Ride subtypes are excluded too so the
    eligible set is exactly "Ride"."""
    for sport_type in ("VirtualRide", "GravelRide", "MountainBikeRide", "EBikeRide"):
        activity = FakeActivity(id=1, sport_type=sport_type)
        assert not is_bike_activity(activity)
        assert extract(activity)["sport_Ride"] == 0.0


def test_latlng_accepted_as_sequence_and_as_object():
    class Point:
        lat = 1.5
        lon = 2.5

    assert extract(FakeActivity(id=1, start_latlng=(1.5, 2.5)))["start_lat"] == pytest.approx(1.5)
    assert extract(FakeActivity(id=1, start_latlng=Point()))["start_lng"] == pytest.approx(2.5)
    assert math.isnan(extract(FakeActivity(id=1, start_latlng=None))["start_lat"])


def test_num_handles_stravalib_custom_types():
    class Distance(float):
        """stravalib's Distance subclasses float."""

    class Quantity:
        magnitude = 12.5

    assert _num(Distance(1234.5)) == pytest.approx(1234.5)
    assert _num(Quantity()) == pytest.approx(12.5)
    assert _num(timedelta(seconds=90)) == pytest.approx(90.0)
    assert math.isnan(_num(None))
    assert math.isnan(_num("not a number"))


def test_gear_id_of_reads_both_shapes():
    assert gear_id_of(FakeActivity(id=1, gear_id="b999")) == "b999"
    assert gear_id_of(FakeActivity(id=1, gear_id=None)) is None
    assert gear_id_of(FakeActivity(id=1, gear_id="  ")) is None


def test_to_matrix_shape(history):
    matrix = to_matrix(history[:10])
    assert matrix.shape == (10, len(FEATURE_NAMES))
