from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.plan_phase5_training_speakers import build_speaker_plan


def test_phase5_protocol_fixes_data_and_device_policy():
    protocol_path = Path(__file__).parents[1] / "configs/phase5/protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))

    assert protocol["phase"] == "5.0"
    assert protocol["objective"]["final_runtime"] == "Windows local microphone application"
    assert protocol["objective"]["not_goal"] == "multi-tenant A100 capacity service"
    assert protocol["device_profiles"]["tuning"]["concurrency_optimization"] is False
    assert protocol["deployment_constraints"]["local_microphone"]["report_network_jitter"] is False

    data = protocol["data_policy"]
    reserved = set(data["reserved_pilot_speakers"])
    reserved.update(data["reserved_full_dev_speakers"])
    reserved.update(data["reserved_full_test_speakers"])
    assert len(reserved) == 20
    assert not reserved.intersection(data.get("training_speakers", []))


def test_speaker_plan_excludes_reserved_and_splits_debug_full():
    archive = [f"S{index:04d}" for index in range(1, 41)]
    reserved = {
        "phase3_pilot": ["S0002", "S0003"],
        "phase3_full_dev": ["S0004", "S0005"],
        "phase3_full_test": ["S0006", "S0007"],
    }

    plan = build_speaker_plan(
        archive,
        reserved=reserved,
        debug_count=8,
        selection_seed="phase5-test",
    )

    debug = plan["tiers"]["debug"]["speakers"]
    full = plan["tiers"]["full"]["speakers"]
    all_reserved = (
        reserved["phase3_pilot"] + reserved["phase3_full_dev"] + reserved["phase3_full_test"]
    )
    assert plan["archive_speaker_count"] == 40
    assert plan["reserved_speaker_count"] == 6
    assert plan["tuning_candidate_count"] == 34
    assert len(debug) == 8
    assert len(full) == 26
    assert not set(debug + full).intersection(all_reserved)
    assert not set(debug).intersection(full)
    assert plan["constraints"]["reserved_speakers_excluded"] is True


def test_speaker_plan_rejects_overlap_and_insufficient_candidates():
    reserved = {
        "phase3_pilot": ["S0001"],
        "phase3_full_dev": ["S0002"],
        "phase3_full_test": ["S0002"],
    }
    with pytest.raises(ValueError, match="overlap"):
        build_speaker_plan(
            ["S0001", "S0002", "S0003"],
            reserved=reserved,
            debug_count=1,
            selection_seed="x",
        )

    with pytest.raises(ValueError, match="insufficient|available"):
        build_speaker_plan(
            ["S0001", "S0002", "S0003"],
            reserved={
                "phase3_pilot": [],
                "phase3_full_dev": ["S0001"],
                "phase3_full_test": ["S0002"],
            },
            debug_count=2,
            selection_seed="x",
        )
