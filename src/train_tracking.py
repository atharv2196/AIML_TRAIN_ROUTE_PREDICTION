"""Simulated train positions; values are not live GNSS measurements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class SimulatedTrain:
    train_id: str
    latitude: float
    longitude: float
    speed_kmph: float
    current_station: str
    destination: str
    timestamp: datetime


def create_train(
    train_id: str,
    station: str,
    destination: str,
    latitude: float,
    longitude: float,
    speed_kmph: float = 80.0,
) -> SimulatedTrain:
    return SimulatedTrain(
        train_id=train_id,
        latitude=latitude,
        longitude=longitude,
        speed_kmph=speed_kmph,
        current_station=station,
        destination=destination,
        timestamp=datetime.now(timezone.utc),
    )


def update_position(train: SimulatedTrain, station: str, latitude: float, longitude: float) -> SimulatedTrain:
    train.current_station = station
    train.latitude = latitude
    train.longitude = longitude
    train.timestamp = datetime.now(timezone.utc)
    return train


def locate_position(train: SimulatedTrain, station_coordinates: dict[str, tuple[float, float]]) -> str:
    """Map a simulated coordinate to the nearest known station."""
    if train.current_station in station_coordinates:
        return train.current_station
    return min(
        station_coordinates,
        key=lambda station: (station_coordinates[station][0] - train.latitude) ** 2
        + (station_coordinates[station][1] - train.longitude) ** 2,
    )
