"""Compare inference interfaces to checkpoint-side YAML without constructing YAML objects."""

from pathlib import Path

import yaml


def differences(expected, actual, path=""):
    result = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if list(expected) != list(actual):
            result.append(f"{path}: keys/order differ")
        for key in expected.keys() & actual.keys():
            result.extend(differences(expected[key], actual[key], f"{path}.{key}"))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            result.append(f"{path}: length {len(expected)} != {len(actual)}")
        else:
            for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                result.extend(differences(left, right, f"{path}[{index}]"))
    elif expected != actual:
        result.append(f"{path}: {expected!r} != {actual!r}")
    return result


def audit_training_config(checkpoint, env_config, agent_config):
    params = Path(checkpoint).resolve().parent / "params"
    paths = [params / "env.yaml", params / "agent.yaml"]
    if not all(path.is_file() for path in paths):
        print(f"[config] Training YAML missing under {params}; observation compatibility NOT verified.", flush=True)
        return {"status": "unverified", "params": str(params)}
    saved_env, saved_agent = [yaml.load(path.read_text(), Loader=yaml.BaseLoader) for path in paths]
    # Canonicalize tuples/slices/functions as inert YAML data on both sides.
    env, agent = [
        yaml.load(yaml.dump(cfg, sort_keys=False), Loader=yaml.BaseLoader) for cfg in (env_config, agent_config)
    ]
    mismatches = []
    inference_overrides = []
    # play_mode intentionally disables training-time observation corruption.
    for group, current in env["observations"].items():
        previous = saved_env["observations"].get(group)
        if (
            isinstance(current, dict)
            and isinstance(previous, dict)
            and previous.get("enable_corruption") == "true"
            and current.get("enable_corruption") == "false"
        ):
            previous["enable_corruption"] = "false"
            inference_overrides.append(f"observations.{group}.enable_corruption: true -> false")
    for key in ("observations", "actions", "decimation"):
        mismatches.extend(differences(saved_env[key], env[key], key))
    mismatches.extend(differences(saved_env["sim"]["dt"], env["sim"]["dt"], "sim.dt"))
    for key in ("obs_groups", "actor", "critic", "clip_actions"):
        mismatches.extend(differences(saved_agent[key], agent[key], f"agent.{key}"))
    if mismatches:
        raise ValueError("Checkpoint observation/action interface mismatch:\n" + "\n".join(mismatches))
    print(
        f"[config] Training observation/action configuration matches. Actor groups: {agent['obs_groups']['actor']}",
        flush=True,
    )
    return {
        "status": "matched",
        "params": str(params),
        "actor_groups": agent["obs_groups"]["actor"],
        "inference_overrides": inference_overrides,
        "scope": "serialized observations/actions, policy architecture, dt; not robot geometry or function source",
    }
