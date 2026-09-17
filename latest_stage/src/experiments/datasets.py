"""Configuration models for declarative experiment datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentScenario:
    name: str
    gap: float
    overlap: float
    target_snr_db: float

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "ExperimentScenario":
        try:
            name = str(data["name"]).strip()
            gap = float(data["gap"])
            overlap = float(data["overlap"])
            target_snr_db = float(data["target_snr_db"])
        except KeyError as exc:
            raise ValueError(f"scenario {index} is missing field: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scenario {index} has an invalid numeric value") from exc

        if not name:
            raise ValueError(f"scenario {index} has an empty name")
        if gap < 0:
            raise ValueError(f"scenario {name}: gap must be non-negative")
        if overlap < 0:
            raise ValueError(f"scenario {name}: overlap must be non-negative")
        return cls(name=name, gap=gap, overlap=overlap, target_snr_db=target_snr_db)


@dataclass(frozen=True)
class ExperimentDatasetConfig:
    target_speaker: str
    interferer_speaker: str
    num_utterances: int
    sample_rate: int
    seeds: tuple[int, ...]
    scenarios: tuple[ExperimentScenario, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentDatasetConfig":
        try:
            target_speaker = str(data["target_speaker"]).strip()
            interferer_speaker = str(data["interferer_speaker"]).strip()
            num_utterances = int(data["num_utterances"])
            sample_rate = int(data["sample_rate"])
            raw_seeds = data["seeds"]
            raw_scenarios = data["scenarios"]
        except KeyError as exc:
            raise ValueError(f"experiment config is missing field: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("experiment config has an invalid scalar value") from exc

        if not target_speaker or not interferer_speaker:
            raise ValueError("target_speaker and interferer_speaker are required")
        if target_speaker == interferer_speaker:
            raise ValueError("target and interferer speakers must be different")
        if num_utterances <= 0:
            raise ValueError("num_utterances must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not isinstance(raw_seeds, list) or not raw_seeds:
            raise ValueError("seeds must be a non-empty list")
        if not isinstance(raw_scenarios, list) or not raw_scenarios:
            raise ValueError("scenarios must be a non-empty list")

        try:
            seeds = tuple(int(seed) for seed in raw_seeds)
            scenarios = tuple(
                ExperimentScenario.from_dict(item, index)
                for index, item in enumerate(raw_scenarios)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid experiment configuration: {exc}") from exc

        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        names = [scenario.name for scenario in scenarios]
        if len(set(names)) != len(names):
            raise ValueError("scenario names must be unique")
        return cls(
            target_speaker=target_speaker,
            interferer_speaker=interferer_speaker,
            num_utterances=num_utterances,
            sample_rate=sample_rate,
            seeds=seeds,
            scenarios=scenarios,
        )
