"""Ensure asynchronous environments start independent packets on reset."""

import json

from ..core.episodes import EpisodePackets


def test_only_reset_environment_rotates_packet(tmp_path):
    with EpisodePackets(tmp_path) as packets:
        for env in (0, 1):
            packets.write_state(env, 0, {"step": 1})
        packets.frame_saved(0)
        packets.write_state(0, 1, {"step": 2})
        packets.write_state(1, 0, {"step": 2})
    summaries = {
        str(path.relative_to(tmp_path)): json.loads(path.read_text()) for path in tmp_path.glob("env_*/*/episode.json")
    }
    assert len(summaries) == 3
    assert summaries["env_000/episode_000000/episode.json"]["end_reason"] == "reset"
    assert summaries["env_000/episode_000000/episode.json"]["frame_count"] == 1
    assert summaries["env_000/episode_000001/episode.json"]["state_count"] == 1
    assert summaries["env_001/episode_000000/episode.json"]["state_count"] == 2
