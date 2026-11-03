from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from argparse import ArgumentParser
import numpy as np
import os
import random
import torch
import torch.optim as optim

from smp_tools import PACKAGE_ROOT
from smp_tools.config_util import load_config, save_config
import smp_tools.anim.motion as motion
from smp_tools.tinymdm.tinymdm_model import TinyMDMModel
from smp_tools.motion_prior_dataset import MotionPriorData
import smp_tools.util.logger as logger

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return

def build_logger(log_file):
    log = logger.Logger()
    log.set_step_key("Iter")
    log.configure_output_file(log_file)
    return log

@torch.no_grad()
def generate(model, dataset, obs_space, config, device, out_motion_dir, enable_ema=False, num_samples=16):
    fps = dataset.control_freq
    num_frames = obs_space.shape[-1] // config["input_channel"]

    if enable_ema:
        gen_samples = model.sample_ema(shape=obs_space.shape, batch_size=num_samples, device=device)
    else:
        gen_samples = model.sample(shape=obs_space.shape, batch_size=num_samples, device=device)

    gen_samples = model.unnormalize(gen_samples.reshape([num_samples, num_frames, -1]))

    os.makedirs(out_motion_dir, exist_ok=True)
    for i, gen_sample in enumerate(gen_samples):
        frames = dataset.convert_sample_to_frames(gen_sample).detach().cpu().numpy()
        sample_motion = motion.Motion(loop_mode=motion.LoopMode.CLAMP, fps=fps, frames=frames)

        out_motion_file = os.path.join(out_motion_dir, f"motion_{i:03}.pkl")
        sample_motion.save(out_motion_file)

        pos_sample = dataset.calc_joint_position_from_frame(frames)[None,...]
        dataset.plot_jnt(jnt_pos=pos_sample, out_path=os.path.join(out_motion_dir, f"anim_{i:03}"))

    return

@torch.no_grad()
def test(cfg_path, model_file, out_dir=None, num_samples=16, device="cuda"):
    fixseed(0)
    assert(out_dir is not None and out_dir != ""), "Must specify --out_dir"
    assert(model_file != ""), "Must specify --model_file"

    config = load_config(cfg_path)
    env_config = load_config(config["env_config"])

    out_motion_dir = os.path.join(out_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_motion_dir, exist_ok=True)

    dataset_env = MotionPriorData(config, device)
    obs_space = dataset_env.get_obs_space()
    num_obs_steps = env_config["num_disc_obs_steps"]
    config["input_dim"] = obs_space.shape[-1]
    config["input_channel"] = int(config["input_dim"] / num_obs_steps)

    priormodel = TinyMDMModel(config, device)
    prior_state_dict = torch.load(model_file, map_location=device)
    priormodel.load_state_dict(prior_state_dict)

    priormodel.eval()
    priormodel.to(device)

    generate(priormodel, dataset_env, obs_space, config, device, out_motion_dir=out_motion_dir,
            enable_ema=priormodel.model_ema, num_samples=num_samples)
    return

def train(cfg_path, out_dir=None, device="cuda"):
    assert out_dir is not None and out_dir != "", "Must specify --out_dir"

    config = load_config(cfg_path)
    env_config = load_config(config["env_config"])

    out_motion_dir = os.path.join(out_dir, "samples")
    out_model_file = os.path.join(out_dir, "model.pt")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_motion_dir, exist_ok=True)

    out_env_config_file = str(Path(out_dir, "env_config.yaml").resolve())
    save_config(env_config, out_env_config_file)

    config["env_config"] = out_env_config_file
    out_config_file = os.path.join(out_dir, "diffusion_config.yaml")

    out_log_file = os.path.join(out_dir, "log.txt")
    log = build_logger(out_log_file)

    batch_size = config["batch_size"]
    num_samples_stat = config.get("num_samples_stat", 10_000)
    output_iter = config.get("output_iter", 2_000)
    grad_clip_norm = config.get("grad_clip_norm", 1.0)

    dataset_env = MotionPriorData(config, device)
    obs_space = dataset_env.get_obs_space()
    num_obs_steps = env_config["num_disc_obs_steps"]

    config["input_dim"] = obs_space.shape[-1]
    config["input_channel"] = int(config["input_dim"] / num_obs_steps)
    save_config(config, out_config_file)
    print(f"Input_channel: {config['input_channel']}")

    samples = dataset_env.fetch_obs_demo(num_samples_stat)
    model = TinyMDMModel(config, device)
    model.update_normalizer(samples)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
    model.to(device)

    model.train()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_params / 1_000_000:.2f} M")

    num_iters = config['num_iterations']
    curr_iters = 0
    loss_sum = 0

    while curr_iters < num_iters:
        samples = dataset_env.fetch_obs_demo(batch_size).clone().detach()
        samples = samples.to(device)
        samples = model.normalize(samples.reshape(batch_size, -1, config["input_channel"])).reshape(batch_size, -1)

        loss = model(samples)
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        if model.model_ema is not False:
            model.ema_dmodel.update()

        if (curr_iters % output_iter == 0 and curr_iters != 0) or (curr_iters == num_iters - 1):
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), out_model_file)
            model.eval()
            generate(model, dataset_env, obs_space, config, device, out_motion_dir=out_motion_dir,
                    enable_ema=model.model_ema, num_samples=16)
            model.train()

            log.log("Iteration", curr_iters, collection="0_Main")
            log.log("Loss", loss_sum / output_iter, collection="0_Main")
            log.print_log()
            log.write_log()

            loss_sum = 0

        curr_iters += 1

    return


def main(argv=None):
    parser = ArgumentParser(
        description="Train or sample an SMP motion prior (TinyMDM).",
        epilog=("Command-line paths are relative to the working directory. "
                "Paths inside YAML files are relative to the containing YAML file."),
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--cfg_path", type=str, default=str(PACKAGE_ROOT / "config/tinymdm_multi_clip.yaml"))
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    if args.mode == "train":
        print("Training new model...")
        train(args.cfg_path, out_dir=args.out_dir, device=args.device)
    else:
        print("Testing model...")
        test(args.cfg_path, args.model_file, out_dir=args.out_dir, device=args.device)


if __name__ == "__main__":
    main()
