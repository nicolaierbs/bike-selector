"""The two-bike case is the common starting point and hits a different code
path in XGBoost's sklearn wrapper (binary objective instead of multi:softprob).
"""

from __future__ import annotations

import pytest

from bike_selector.model import BikeClassifier

from .conftest import COMMUTER, ROAD, FakeActivity, make_history


@pytest.fixture
def two_bike_history():
    return [a for a in make_history() if a.gear_id in {ROAD.id, COMMUTER.id}]


def test_trains_and_predicts_with_only_two_bikes(settings, two_bike_history, gateway):
    classifier = BikeClassifier(settings)
    report = classifier.train(two_bike_history, bike_names=gateway.bike_names())

    assert report.n_classes == 2
    predictions = classifier.predict(FakeActivity(id=1))
    assert len(predictions) == 2
    assert sum(p.probability for p in predictions) == pytest.approx(1.0, abs=1e-5)
    assert {p.gear_id for p in predictions} == {ROAD.id, COMMUTER.id}


def test_two_bike_model_round_trips(settings, two_bike_history, gateway):
    classifier = BikeClassifier(settings)
    classifier.train(two_bike_history, bike_names=gateway.bike_names())
    classifier.save()

    reloaded = BikeClassifier(settings)
    assert reloaded.load() is True
    assert len(reloaded.predict(FakeActivity(id=1))) == 2
