from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from bike_selector.model import BikeClassifier, InsufficientData

from .conftest import COMMUTER, GRAVEL, ROAD, FakeActivity, make_history


@pytest.fixture
def trained(settings, history, gateway) -> BikeClassifier:
    classifier = BikeClassifier(settings)
    classifier.train(history, bike_names=gateway.bike_names())
    return classifier


def test_training_report(trained):
    report = trained.report
    assert report is not None
    assert report.n_classes == 3
    assert report.n_samples == 120
    assert set(report.class_counts) == {ROAD.name, GRAVEL.name, COMMUTER.name}


def test_cross_validated_accuracy_beats_baseline(trained):
    report = trained.report
    assert report.cv_accuracy is not None
    assert report.baseline_accuracy is not None
    assert report.cv_accuracy > report.baseline_accuracy
    # The synthetic bikes are well separated; anything below this means the
    # feature pipeline has regressed.
    assert report.cv_accuracy > 0.85


def test_probabilities_sum_to_one_and_are_sorted(trained):
    predictions = trained.predict(FakeActivity(id=1))
    assert len(predictions) == 3
    assert sum(p.probability for p in predictions) == pytest.approx(1.0, abs=1e-5)
    assert predictions == sorted(predictions, key=lambda p: -p.probability)


def test_recognises_a_commute(trained):
    commute = FakeActivity(
        id=1,
        distance=10_500,
        moving_time=timedelta(minutes=27),
        elapsed_time=timedelta(minutes=30),
        average_speed=6.3,
        max_speed=9.8,
        total_elevation_gain=55,
        average_watts=None,
        max_watts=None,
        weighted_average_watts=None,
        kilojoules=None,
        device_watts=False,
        average_cadence=None,
        has_heartrate=False,
        average_heartrate=None,
        max_heartrate=None,
        commute=True,
        start_date_local=datetime(2026, 5, 4, 8, 15),
    )
    assert trained.predict(commute)[0].gear_id == COMMUTER.id


def test_recognises_a_gravel_ride(trained):
    gravel = FakeActivity(
        id=2,
        sport_type="Ride",
        distance=56_000,
        moving_time=timedelta(minutes=165),
        elapsed_time=timedelta(minutes=190),
        average_speed=5.6,
        max_speed=12.2,
        total_elevation_gain=950,
        average_watts=172,
        weighted_average_watts=191,
        average_cadence=77,
    )
    assert trained.predict(gravel)[0].gear_id == GRAVEL.id


def test_unlabelled_rides_are_excluded_from_training(settings, gateway):
    history = make_history()
    for activity in history[:30]:
        activity.gear_id = None
    classifier = BikeClassifier(settings)
    report = classifier.train(history, bike_names=gateway.bike_names())
    assert report.n_samples == 90


def test_self_labelled_rides_are_excluded(settings, history, gateway):
    """Rides this app labelled must not be fed back in as ground truth."""
    excluded = {a.id for a in history[:15]}
    classifier = BikeClassifier(settings)
    report = classifier.train(history, bike_names=gateway.bike_names(), exclude_ids=excluded)
    assert report.n_samples == 105


def test_rare_bikes_are_dropped(settings, gateway):
    history = make_history()
    rare = FakeActivity(id=99_999, gear_id="b444")
    classifier = BikeClassifier(settings)
    report = classifier.train([*history, rare], bike_names=gateway.bike_names())
    assert report.n_classes == 3
    assert "b444" not in classifier.labels
    assert report.dropped_bikes == {"b444": 1}


def test_single_bike_raises(settings, gateway):
    history = [a for a in make_history() if a.gear_id == ROAD.id]
    with pytest.raises(InsufficientData):
        BikeClassifier(settings).train(history, bike_names=gateway.bike_names())


def test_no_labelled_rides_raises(settings):
    history = make_history()
    for activity in history:
        activity.gear_id = None
    with pytest.raises(InsufficientData, match="No rides with a bike"):
        BikeClassifier(settings).train(history)


def test_round_trip_through_disk(settings, trained):
    path = trained.save()
    assert path.exists()

    reloaded = BikeClassifier(settings)
    assert reloaded.load() is True
    assert reloaded.labels == trained.labels
    assert reloaded.bike_names == trained.bike_names

    before = trained.predict(FakeActivity(id=1))
    after = reloaded.predict(FakeActivity(id=1))
    assert [p.gear_id for p in before] == [p.gear_id for p in after]
    assert before[0].probability == pytest.approx(after[0].probability)


def test_stale_cache_is_rejected(settings, trained, monkeypatch):
    trained.save()
    reloaded = BikeClassifier(settings)
    monkeypatch.setattr(type(reloaded), "age_hours", property(lambda self: 999.0))
    assert reloaded.load() is False


def test_load_returns_false_without_a_cache(settings):
    assert BikeClassifier(settings).load() is False


def test_prediction_formatting():
    from bike_selector.model import Prediction

    assert Prediction("b1", "Canyon", 0.8712).format() == "Canyon 87%"
    assert Prediction("b1", "Canyon", 0.8712).format(digits=1) == "Canyon 87.1%"
