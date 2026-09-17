from __future__ import annotations

import pytest

from src.experiments.datasets import ExperimentDatasetConfig


def _config(**overrides):
    data = {
        "target_speaker": "speaker1",
        "interferer_speaker": "speaker2",
        "num_utterances": 4,
        "sample_rate": 16000,
        "seeds": [42, 43],
        "scenarios": [
            {
                "name": "alternate_snr8",
                "gap": 1.0,
                "overlap": 0.0,
                "target_snr_db": 8.0,
            }
        ],
    }
    data.update(overrides)
    return data


def test_config_parses_scenarios_and_seeds():
    config = ExperimentDatasetConfig.from_dict(_config())
    assert config.target_speaker == "speaker1"
    assert config.seeds == (42, 43)
    assert config.scenarios[0].name == "alternate_snr8"


def test_config_rejects_same_target_and_interferer():
    with pytest.raises(ValueError, match="must be different"):
        ExperimentDatasetConfig.from_dict(_config(interferer_speaker="speaker1"))


def test_config_rejects_duplicate_scenario_names():
    scenario = {
        "name": "same",
        "gap": 1.0,
        "overlap": 0.0,
        "target_snr_db": 8.0,
    }
    with pytest.raises(ValueError, match="scenario names must be unique"):
        ExperimentDatasetConfig.from_dict(_config(scenarios=[scenario, scenario]))
