from __future__ import annotations

import pytest

from bike_selector.model import BikeClassifier, Prediction
from bike_selector.service import BikeSelectorService

from .conftest import COMMUTER, GRAVEL, ROAD, FakeActivity, FakeGateway, make_history


@pytest.fixture
def service(settings, gateway) -> BikeSelectorService:
    return BikeSelectorService(gateway=gateway, classifier=BikeClassifier(settings), settings=settings)


PREDICTIONS = [
    Prediction(ROAD.id, ROAD.name, 0.87),
    Prediction(GRAVEL.id, GRAVEL.name, 0.11),
    Prediction(COMMUTER.id, COMMUTER.name, 0.02),
]

PREDICTIONS_UNSURE = [
    Prediction(ROAD.id, ROAD.name, 0.44),
    Prediction(GRAVEL.id, GRAVEL.name, 0.38),
    Prediction(COMMUTER.id, COMMUTER.name, 0.18),
]


# ------------------------------------------------------------- descriptions


def test_description_appended_below_existing_text(service):
    result = service.build_description("Lovely morning loop.", PREDICTIONS)
    assert result.startswith("Lovely morning loop.")
    assert "🥇 Canyon Endurace — 87%" in result
    assert "🥈 Cube Nuroad — 11%" in result
    assert "🥉 Rose Commuter — 2%" in result
    assert "https://www.erbs.eu/bikeselector/" in result
    assert result.count("[bike-selector]") == 2  # wraps the block, start and end
    assert result.endswith("[bike-selector]")


def test_description_created_when_empty(service):
    assert service.build_description(None, PREDICTIONS).startswith("[bike-selector]")
    assert "🚲 Bike guess" in service.build_description("", PREDICTIONS)


def test_rerunning_replaces_the_old_block_instead_of_stacking(service):
    first = service.build_description("Ride notes", PREDICTIONS)
    second = service.build_description(first, PREDICTIONS)
    assert second.count("🚲 Bike guess") == 1
    assert second.count("[bike-selector]") == 2
    assert second.startswith("Ride notes")


def test_legacy_single_line_format_is_replaced_not_stacked(service):
    legacy = "Ride notes\n\nBike guess: Canyon Endurace 90% [bike-selector]"
    result = service.build_description(legacy, PREDICTIONS)
    assert "Bike guess: Canyon Endurace 90%" not in result
    assert result.count("[bike-selector]") == 2


def test_probability_count_is_configurable(service):
    service.settings.max_probabilities_shown = 2
    line = service.build_description(None, PREDICTIONS)
    assert "Rose Commuter" not in line


# ---------------------------------------------------------------------- title


def test_default_strava_title_gets_renamed(service):
    activity = FakeActivity(id=580, gear_id=None, name="Morning Ride")
    result = service.build_title(activity, PREDICTIONS[0])
    assert result is not None
    assert result != "Morning Ride"


def test_custom_title_is_left_alone(service):
    activity = FakeActivity(id=581, gear_id=None, name="Sunday loop with the club")
    assert service.build_title(activity, PREDICTIONS[0]) is None


def test_custom_title_is_overwritten_when_forced(service):
    activity = FakeActivity(id=582, gear_id=None, name="Sunday loop with the club")
    assert service.build_title(activity, PREDICTIONS[0], force=True) is not None


def test_custom_title_is_overwritten_when_configured(service):
    service.settings.overwrite_existing_title = True
    activity = FakeActivity(id=583, gear_id=None, name="Sunday loop with the club")
    assert service.build_title(activity, PREDICTIONS[0]) is not None


def test_title_rename_disabled_by_setting(service):
    service.settings.rename_title = False
    activity = FakeActivity(id=584, gear_id=None, name="Morning Ride")
    assert service.build_title(activity, PREDICTIONS[0]) is None


def test_processing_renames_the_default_title(service, gateway):
    gateway.by_id[585] = FakeActivity(id=585, gear_id=None, name="Morning Ride")
    outcome = service.process_activity(585)
    assert outcome.status == "updated"
    assert outcome.new_title
    assert gateway.updates[0]["name"] == outcome.new_title
    assert gateway.by_id[585].name == outcome.new_title


def test_processing_leaves_a_custom_title_alone(service, gateway):
    gateway.by_id[586] = FakeActivity(id=586, gear_id=None, name="Race day with Sam")
    outcome = service.process_activity(586)
    assert outcome.new_title is None
    assert gateway.updates[0]["name"] is None
    assert gateway.by_id[586].name == "Race day with Sam"


def test_dry_run_reports_the_title_without_writing(service, gateway):
    gateway.by_id[587] = FakeActivity(id=587, gear_id=None, name="Morning Ride")
    outcome = service.process_activity(587, dry_run=True)
    assert outcome.status == "dry-run"
    assert outcome.new_title
    assert gateway.updates == []
    assert gateway.by_id[587].name == "Morning Ride"


# ----------------------------------------------------------------- pipeline


def test_updates_gear_and_description(service, gateway):
    target = FakeActivity(id=555, gear_id=None, description="Commute home")
    gateway.by_id[555] = target

    outcome = service.process_activity(555)

    assert outcome.status == "updated"
    assert outcome.chosen_gear_id in {ROAD.id, GRAVEL.id, COMMUTER.id}
    assert len(gateway.updates) == 1
    update = gateway.updates[0]
    assert update["gear_id"] == outcome.chosen_gear_id
    assert "🚲 Bike guess" in update["description"]
    assert update["description"].startswith("Commute home")


def test_probabilities_reported_for_every_bike(service, gateway):
    gateway.by_id[556] = FakeActivity(id=556, gear_id=None)
    outcome = service.process_activity(556)
    assert outcome.probabilities is not None
    assert len(outcome.probabilities) == 3
    assert sum(p["probability"] for p in outcome.probabilities) == pytest.approx(1.0, abs=1e-3)


def test_runs_are_skipped(service, gateway):
    gateway.by_id[557] = FakeActivity(id=557, sport_type="Run", gear_id=None)
    outcome = service.process_activity(557)
    assert outcome.status == "skipped"
    assert "not a bike activity" in outcome.detail
    assert gateway.updates == []


def test_manual_activities_are_skipped(service, gateway):
    gateway.by_id[558] = FakeActivity(id=558, manual=True, gear_id=None)
    assert service.process_activity(558).status == "skipped"


def test_trainer_rides_skipped_when_configured(service, gateway):
    service.settings.skip_trainer = True
    gateway.by_id[559] = FakeActivity(id=559, trainer=True, gear_id=None)
    assert service.process_activity(559).status == "skipped"


def test_already_annotated_activity_is_left_alone(service, gateway):
    gateway.by_id[560] = FakeActivity(
        id=560, gear_id=None, description="Nice ride\n\nBike guess: X 90% [bike-selector]"
    )
    outcome = service.process_activity(560)
    assert outcome.status == "skipped"
    assert gateway.updates == []


def test_force_overrides_the_annotation_guard(service, gateway):
    gateway.by_id[561] = FakeActivity(
        id=561, gear_id=None, description="Nice ride\n\nBike guess: X 90% [bike-selector]"
    )
    assert service.process_activity(561, force=True).status == "updated"


def test_existing_gear_respected_when_overwrite_disabled(service, gateway):
    service.settings.overwrite_existing_gear = False
    gateway.by_id[562] = FakeActivity(id=562, gear_id=GRAVEL.id)
    outcome = service.process_activity(562)
    assert outcome.status == "skipped"
    assert gateway.updates == []


def test_low_confidence_annotates_without_assigning(service, gateway, monkeypatch):
    """Below MIN_CONFIDENCE we still report the odds but leave the gear alone."""
    service.settings.min_confidence = 0.8
    monkeypatch.setattr(
        service.classifier, "predict", lambda _activity: list(PREDICTIONS_UNSURE)
    )
    monkeypatch.setattr(service, "ensure_model", lambda **_: service.classifier)

    gateway.by_id[563] = FakeActivity(id=563, gear_id=None)
    outcome = service.process_activity(563)

    assert outcome.status == "updated"
    assert gateway.updates[0]["gear_id"] is None
    assert "🚲 Bike guess" in gateway.updates[0]["description"]


def test_confident_prediction_does_assign(service, gateway, monkeypatch):
    service.settings.min_confidence = 0.8
    monkeypatch.setattr(service.classifier, "predict", lambda _activity: list(PREDICTIONS))
    monkeypatch.setattr(service, "ensure_model", lambda **_: service.classifier)

    gateway.by_id[566] = FakeActivity(id=566, gear_id=None)
    service.process_activity(566)

    assert gateway.updates[0]["gear_id"] == ROAD.id


def test_dry_run_writes_nothing(service, gateway):
    gateway.by_id[564] = FakeActivity(id=564, gear_id=None)
    outcome = service.process_activity(564, dry_run=True)
    assert outcome.status == "dry-run"
    assert outcome.probabilities is not None
    assert gateway.updates == []


def test_auto_assigned_ids_are_remembered_for_future_training(service, gateway):
    gateway.by_id[565] = FakeActivity(id=565, gear_id=None)
    service.process_activity(565)
    assert 565 in service._auto_labelled
    assert service.settings.auto_labelled_path.exists()


# ------------------------------------------------------------------- events


def test_create_event_is_processed(service, gateway):
    gateway.by_id[600] = FakeActivity(id=600, gear_id=None)
    outcome = service.handle_event(
        {"object_type": "activity", "aspect_type": "create", "object_id": 600}
    )
    assert outcome is not None and outcome.status == "updated"


@pytest.mark.parametrize(
    "event",
    [
        {"object_type": "athlete", "aspect_type": "create", "object_id": 1},
        {"object_type": "activity", "aspect_type": "update", "object_id": 1},
        {"object_type": "activity", "aspect_type": "delete", "object_id": 1},
        {"object_type": "activity", "aspect_type": "create"},
    ],
)
def test_irrelevant_events_are_ignored(service, event):
    assert service.handle_event(event) is None


# ----------------------------------------------------------------- backfill


def test_backfill_only_touches_rides_without_gear(settings):
    history = make_history()
    for activity in history[:6]:
        activity.gear_id = None
    gateway = FakeGateway(history)
    service = BikeSelectorService(
        gateway=gateway, classifier=BikeClassifier(settings), settings=settings
    )

    outcomes = service.backfill(days=3650, limit=200, dry_run=True)

    assert len(outcomes) == 6
    assert all(o.status == "dry-run" for o in outcomes)
    assert gateway.updates == []


def test_model_is_trained_lazily_and_reused(service, gateway):
    assert not service.classifier.is_trained
    gateway.by_id[700] = FakeActivity(id=700, gear_id=None)
    service.process_activity(700, dry_run=True)
    assert service.classifier.is_trained
    report = service.classifier.report

    gateway.by_id[701] = FakeActivity(id=701, gear_id=None)
    service.process_activity(701, dry_run=True)
    assert service.classifier.report is report  # no retrain
