"""Finite, stratified LP-ACRL evaluation using the checkpoint's saved configurations.

Example:
python -m Qmini.tasks.quadruped.as2w_lp_cc.evaluate --checkpoint RUN/model_1499.pt \
    --num_envs 32 --episodes 3 --output evaluation.json --viz none
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from .env.as2w_env_cfg import AS2WFlatEnvCfg, AS2WRoughEnvCfg
from .env.multi_critic_env_cfg import AS2WFlatMultiCriticEnvCfg, AS2WRoughMultiCriticEnvCfg


class ConfigLoader(yaml.SafeLoader):
    """Load only the tuple/slice tags emitted by Isaac Lab config serialization."""


ConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple',
                             lambda loader, node: tuple(loader.construct_sequence(node, deep=True)))
ConfigLoader.add_constructor('tag:yaml.org,2002:python/object/apply:builtins.slice',
                             lambda loader, node: slice(*loader.construct_sequence(node, deep=True)))


def load_config(path):
    return yaml.load(Path(path).read_text(), Loader=ConfigLoader)


def restored_configs(checkpoint):
    params = Path(checkpoint).parent/'params'
    saved_env = load_config(params/'env.yaml')
    agent = load_config(params/'agent.yaml')
    flat = saved_env['scene']['terrain']['terrain_type'] == 'plane'
    multi = 'critic_groups' in agent['algorithm']
    cls = {(False, False): AS2WRoughEnvCfg, (True, False): AS2WFlatEnvCfg,
           (False, True): AS2WRoughMultiCriticEnvCfg, (True, True): AS2WFlatMultiCriticEnvCfg}[flat, multi]
    cfg = cls()
    cfg.sim.physics = cfg.sim.physics.isaacsim_physx
    cfg.from_dict(saved_env)
    # This evaluator is deterministic actor inference with uniform task coverage.
    cfg.lp_acrl.adaptive = False
    task = f"Qmini-Velocity-{'Flat' if flat else 'Rough'}-AS2W-LP-ACRL" + ('-MultiCritic' if multi else '')
    return cfg, agent, task


@torch.inference_mode()
def evaluate(env, policy, task_ids, episodes):
    raw = env.unwrapped
    controller = raw.lp_curriculum
    controller.record_evaluation = True
    records = []
    for offset in range(0, len(task_ids), env.num_envs):
        batch = task_ids[offset:offset+env.num_envs]
        padded = batch + [batch[-1]]*(env.num_envs-len(batch))
        controller.forced_task_ids = torch.tensor(padded, device=env.device, dtype=torch.long)
        for _ in range(episodes):
            obs, _ = env.reset()
            controller.evaluation_records.clear()
            seen = set()
            for _ in range(raw.max_episode_length+1):
                obs, _, _, _ = env.step(policy(obs))
                for record in controller.evaluation_records:
                    slot = record['env_id']
                    if slot < len(batch) and slot not in seen:
                        records.append(record)
                        seen.add(slot)
                controller.evaluation_records.clear()
                if len(seen) == len(batch):
                    break
            if len(seen) != len(batch):
                raise RuntimeError("Evaluation did not finish all task episodes within the horizon")
        print(f"Evaluated {min(offset+len(batch), len(task_ids))}/{len(task_ids)} tasks", flush=True)
    rows = []
    for task_id in task_ids:
        data = [r for r in records if r['task_id'] == task_id]
        mean_epte = [sum(r['epte'][i] for r in data)/len(data) for i in range(3)]
        alive = sum(r['alive_steps'] for r in data)/len(data)
        terrain, level, vx, yaw = controller.sampler.table[task_id].tolist()
        cfg = controller.sampler.cfg
        rows.append(dict(task_id=task_id, terrain=cfg.terrain_names[terrain], level=level,
                         vx_interval=cfg.vx_edges[vx:vx+2], yaw_interval=cfg.yaw_edges[yaw:yaw+2],
                         episodes=len(data), mean_return=sum(r['return'] for r in data)/len(data),
                         mean_alive_steps=alive, epte_x_y_yaw=mean_epte,
                         crossed_tile_fraction=sum(r['crossed_tile'] for r in data)/len(data),
                         success=alive > .9*raw.max_episode_length and mean_epte[0] < .3 and mean_epte[2] < .3))
    return dict(horizon=raw.max_episode_length, task_count=len(rows), episodes_per_task=episodes,
                success_fraction=sum(r['success'] for r in rows)/len(rows), tasks=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--num_envs', type=int, default=32)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--task_ids', type=int, nargs='+', help='Default: every task, uniformly evaluated')
    parser.add_argument('--output', type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes < 1:
        parser.error('num_envs and episodes must be positive')
    cfg, agent_cfg, task = restored_configs(args.checkpoint)
    cfg.scene.num_envs = args.num_envs
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make(task, cfg=cfg)
        try:
            env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.get('clip_actions'))
            runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=env.device)
            runner.load(str(args.checkpoint), map_location=env.device,
                        load_cfg={'actor': True, 'critic': True, 'optimizer': False, 'iteration': False})
            ids = args.task_ids or list(range(raw.unwrapped.lp_curriculum.sampler.num_tasks))
            if len(ids) != len(set(ids)) or min(ids) < 0 or max(ids) >= raw.unwrapped.lp_curriculum.sampler.num_tasks:
                parser.error('task_ids must be distinct valid task indices')
            summary = evaluate(env, runner.get_inference_policy(), ids, args.episodes)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
            print(f"Saved evaluation to {args.output}", flush=True)
        finally:
            raw.close()


if __name__ == '__main__':
    main()
