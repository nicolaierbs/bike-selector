"""Synthetic Strava objects so the suite runs without network or credentials."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from bike_selector.config import Settings


@dataclass
class FakeGear:
    id: str
    name: str
    primary: bool = False
    retired: bool = False
    distance: float = 0.0


@dataclass
class FakeAthlete:
    id: int = 42
    bikes: list[FakeGear] = field(default_factory=list)


@dataclass
class FakeActivity:
    """Mimics the attribute surface of stravalib's SummaryActivity/DetailedActivity."""

    id: int
    sport_type: str = "Ride"
    distance: float = 30_000.0
    moving_time: timedelta = timedelta(hours=1)
    elapsed_time: timedelta = timedelta(hours=1, minutes=5)
    average_speed: float = 8.3
    max_speed: float = 14.0
    total_elevation_gain: float = 250.0
    elev_high: float | None = 400.0
    elev_low: float | None = 150.0
    average_watts: float | None = 180.0
    max_watts: float | None = 600.0
    weighted_average_watts: float | None = 195.0
    kilojoules: float | None = 650.0
    device_watts: bool | None = True
    has_heartrate: bool = True
    average_heartrate: float | None = 140.0
    max_heartrate: float | None = 175.0
    average_cadence: float | None = 85.0
    average_temp: float | None = 18.0
    trainer: bool = False
    commute: bool = False
    manual: bool = False
    start_latlng: tuple[float, float] | None = (49.87, 8.65)
    start_date: datetime = field(
        default_factory=lambda: datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc)
    )
    start_date_local: datetime = field(default_factory=lambda: datetime(2026, 5, 1, 11, 0))
    achievement_count: int = 3
    pr_count: int = 1
    athlete_count: int = 1
    gear_id: str | None = "b111"
    description: str | None = None


ROAD = FakeGear(id="b111", name="Canyon Endurace", primary=True, distance=42_000_000)
GRAVEL = FakeGear(id="b222", name="Cube Nuroad", distance=8_000_000)
COMMUTER = FakeGear(id="b333", name="Rose Commuter", distance=5_000_000)


def make_history(n_per_bike: int = 40, seed: int = 7) -> list[FakeActivity]:
    """Three bikes with distinct but overlapping signatures.

    Road: long, fast, powered, hilly.
    Gravel: slower, powered, very hilly.
    Commuter: short flat hops with no power meter and no HR.

    All tagged sport_type "Ride" — that is the only sport type the app trains
    or predicts on (VirtualRide and other Ride subtypes are excluded).

    Every field carries noise on purpose. A zero-variance field would become a
    perfect giveaway and the tests would stop telling us anything about whether
    the real features discriminate.
    """
    rng = random.Random(seed)
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    activities: list[FakeActivity] = []
    next_id = 1000

    def local(offset: timedelta, hour_low: float, hour_high: float) -> datetime:
        return (base - offset).replace(tzinfo=None) + timedelta(
            hours=rng.uniform(hour_low, hour_high)
        )

    for index in range(n_per_bike):
        offset = timedelta(days=index * 2)

        activities.append(
            FakeActivity(
                id=next_id,
                sport_type="Ride",
                gear_id=ROAD.id,
                distance=rng.gauss(70_000, 12_000),
                moving_time=timedelta(minutes=rng.gauss(150, 25)),
                elapsed_time=timedelta(minutes=rng.gauss(165, 25)),
                average_speed=rng.gauss(8.0, 0.5),
                max_speed=rng.gauss(16.0, 1.5),
                total_elevation_gain=rng.gauss(700, 150),
                average_watts=rng.gauss(200, 20),
                weighted_average_watts=rng.gauss(215, 20),
                device_watts=True,
                average_cadence=rng.gauss(88, 4),
                has_heartrate=True,
                average_heartrate=rng.gauss(145, 8),
                start_date=base - offset,
                start_date_local=local(offset, 8, 15),
            )
        )
        next_id += 1

        activities.append(
            FakeActivity(
                id=next_id,
                sport_type="Ride",
                gear_id=GRAVEL.id,
                distance=rng.gauss(55_000, 10_000),
                moving_time=timedelta(minutes=rng.gauss(160, 30)),
                elapsed_time=timedelta(minutes=rng.gauss(185, 30)),
                average_speed=rng.gauss(5.6, 0.5),
                max_speed=rng.gauss(12.0, 1.5),
                total_elevation_gain=rng.gauss(900, 200),
                average_watts=rng.gauss(170, 20),
                weighted_average_watts=rng.gauss(190, 20),
                device_watts=True,
                average_cadence=rng.gauss(78, 5),
                has_heartrate=True,
                average_heartrate=rng.gauss(150, 8),
                start_date=base - offset,
                start_date_local=local(offset, 8, 16),
            )
        )
        next_id += 1

        activities.append(
            FakeActivity(
                id=next_id,
                sport_type="Ride",
                gear_id=COMMUTER.id,
                distance=rng.gauss(11_000, 1_500),
                moving_time=timedelta(minutes=rng.gauss(28, 4)),
                elapsed_time=timedelta(minutes=rng.gauss(31, 4)),
                average_speed=rng.gauss(6.2, 0.4),
                max_speed=rng.gauss(10.0, 1.0),
                total_elevation_gain=rng.gauss(60, 20),
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
                start_date=base - offset,
                start_date_local=local(offset, 7, 19),
            )
        )
        next_id += 1

    return activities


class FakeGateway:
    """Stands in for StravaGateway; records writes instead of calling Strava."""

    def __init__(self, activities: list[FakeActivity] | None = None) -> None:
        self.activities = activities if activities is not None else make_history()
        self.by_id = {a.id: a for a in self.activities}
        self.updates: list[dict[str, Any]] = []
        self.gear = [ROAD, GRAVEL, COMMUTER]

    def bikes(self) -> list[FakeGear]:
        return self.gear

    def bike_names(self) -> dict[str, str]:
        return {g.id: g.name for g in self.gear}

    def athlete_id(self) -> int:
        return 42

    def recent_activities(self, limit: int | None = None) -> list[FakeActivity]:
        return self.activities[: limit or len(self.activities)]

    def activity(self, activity_id: int) -> FakeActivity:
        return self.by_id[int(activity_id)]

    def update_activity(
        self, activity_id: int, *, gear_id: str | None = None, description: str | None = None
    ) -> None:
        self.updates.append(
            {"activity_id": activity_id, "gear_id": gear_id, "description": description}
        )
        activity = self.by_id[int(activity_id)]
        if gear_id is not None:
            activity.gear_id = gear_id
        if description is not None:
            activity.description = description


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        strava_client_id=1,
        strava_client_secret="secret",
        strava_refresh_token="refresh",
        state_dir=tmp_path / "state",
        min_rides_per_bike=5,
        keepalive_minutes=0,
        public_base_url="",
        _env_file=None,
    )


@pytest.fixture
def history() -> list[FakeActivity]:
    return make_history()


@pytest.fixture
def gateway(history) -> FakeGateway:
    return FakeGateway(history)
