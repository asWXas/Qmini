"""One independently readable data directory per environment episode."""

import json


class EpisodePackets:
    def __init__(self, output):
        self.output = output
        self.active = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        for index in list(self.active):
            self.finish(index, "interrupted" if exc_type else "run_end")

    def write_state(self, index, episode, record):
        if index in self.active and self.active[index]["episode_id"] != episode:
            self.finish(index, "reset")
        if index not in self.active:
            folder = self.output / f"env_{index:03d}" / f"episode_{episode:06d}"
            folder.mkdir(parents=True, exist_ok=False)
            self.active[index] = {
                "folder": folder,
                "stream": (folder / "states.jsonl").open("w"),
                "env_id": index,
                "episode_id": episode,
                "start_step": record["step"],
                "frame_count": 0,
                "state_count": 0,
                "end_step": record["step"],
            }
            self._summary(index, "recording")
        packet = self.active[index]
        packet["stream"].write(json.dumps(record, allow_nan=False) + "\n")
        packet["stream"].flush()
        packet["state_count"] += 1
        packet["end_step"] = record["step"]

    def frame_saved(self, index):
        self.active[index]["frame_count"] += 1

    def _summary(self, index, reason):
        packet = self.active[index]
        summary = {key: value for key, value in packet.items() if key not in ("folder", "stream")}
        summary["end_reason"] = reason
        (packet["folder"] / "episode.json").write_text(json.dumps(summary, indent=2) + "\n")

    def finish(self, index, reason):
        self.active[index]["stream"].close()
        self._summary(index, reason)
        del self.active[index]
