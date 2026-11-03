"""Exercise the standalone training path with temporary synthetic motion data."""

import importlib.abc
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class RejectRepositoryImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mimickit', 'anim', 'util', 'learning', 'tools', 'envs', 'engines'}:
            raise ImportError(f'Standalone test detected an external repository import: {fullname}')
        return None


sys.meta_path.insert(0, RejectRepositoryImports())
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import torch
import yaml

from smp_tools import PACKAGE_ROOT
from smp_tools.anim.mjcf_char_model import MJCFCharModel
from smp_tools.anim.motion import LoopMode, Motion, load_motion
from smp_tools.anim.urdf_char_model import URDFCharModel
from smp_tools.anim.usd_char_model import USDCharModel
from smp_tools.config_util import load_config
from smp_tools.motion_prior_dataset import MotionPriorData
from smp_tools.train_prior import test, train


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    character = MJCFCharModel('cpu')
    character.load(str(PACKAGE_ROOT / 'data/assets/humanoid/humanoid.xml'))
    usd_character = USDCharModel('cpu')
    usd_character.load(str(PACKAGE_ROOT / 'data/assets/humanoid/humanoid.usd'))
    assert usd_character.get_dof_size() == character.get_dof_size()

    with tempfile.TemporaryDirectory(prefix='smp-standalone-') as tmp:
        work = Path(tmp) / 'original'
        work.mkdir()
        (work / 'assets').mkdir()
        shutil.copyfile(PACKAGE_ROOT / 'data/assets/humanoid/humanoid.xml', work / 'assets/humanoid.xml')
        urdf = work / 'character.urdf'
        urdf.write_text('''<robot name="test"><link name="root"/><link name="limb"/>
          <joint name="hinge" type="revolute"><parent link="root"/><child link="limb"/>
          <origin xyz="0 0 -0.4" rpy="0 0 0"/><axis xyz="0 1 0"/>
          <limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>''')
        urdf_character = URDFCharModel('cpu')
        urdf_character.load(str(urdf))
        assert urdf_character.get_dof_size() == 1
        print('PASS: bundled MJCF / USD and URDF readers work without a simulator')

        frames = np.zeros((12, 6 + character.get_dof_size()), dtype=np.float32)
        frames[:, 0] = np.linspace(0, 0.3, len(frames))
        frames[:, 2] = 1.0
        frames[:, 6] = np.linspace(-0.2, 0.2, len(frames))
        Motion(LoopMode.CLAMP, 30, frames).save(str(work / 'motion.pkl'))
        (work / 'motions.yaml').write_text(yaml.safe_dump({'motions': [{'file': 'motion.pkl', 'weight': 1.0}]}))

        obs = load_config(PACKAGE_ROOT / 'config/humanoid_local_obs.yaml')
        obs.update(char_file='assets/humanoid.xml', num_disc_obs_steps=3)
        (work / 'obs.yaml').write_text(yaml.safe_dump(obs))
        config = load_config(PACKAGE_ROOT / 'config/tinymdm_multi_clip.yaml')
        config.update(env_config='obs.yaml', motion_file='motions.yaml', T=2,
                      batch_size=2, num_iterations=1, num_samples_stat=4, output_iter=1,
                      num_layers=1, num_attention_heads=1, attention_head_dim=8,
                      model_ema_steps=1, model_ema_update_after=0)
        config_file = work / 'config.yaml'
        config_file.write_text(yaml.safe_dump(config))

        # Absolute top-level config, relative nested paths; CWD is unrelated.
        train(str(config_file), str(work / 'output'), device='cpu')
        output = work / 'output'
        weights = torch.load(output / 'model.pt', map_location='cpu', weights_only=True)
        assert weights['obs_normalizer._count'].item() > 0
        assert all(torch.isfinite(value).all() for value in weights.values())
        assert 'Loss' in (output / 'log.txt').read_text()
        assert len(list((output / 'samples').glob('*.pkl'))) == 16
        assert len(list((output / 'samples').glob('*.gif'))) == 16
        for name in ('diffusion_config.yaml', 'env_config.yaml'):
            assert str(work) not in (output / name).read_text()
        print('PASS: data loading, CPU update, EMA, normalization and checkpoint output')

        relocated = Path(tmp) / 'relocated'
        shutil.move(str(work), relocated)
        output = relocated / 'output'
        test(str(output / 'diffusion_config.yaml'), str(output / 'model.pt'),
             str(relocated / 'sampled'), num_samples=2, device='cpu')
        for path in (relocated / 'sampled/samples').glob('*.pkl'):
            sample = load_motion(str(path))
            assert sample.frames.shape == (3, frames.shape[-1])
            assert np.isfinite(sample.frames).all()
        assert len(list((relocated / 'sampled/samples').glob('*.gif'))) == 2
        print('PASS: relocated configuration, checkpoint reload and generated motions / GIFs')

        config = load_config(relocated / 'config.yaml')
        config['motion_file'] = str(relocated / 'motion.pkl')
        data = MotionPriorData(config, 'cpu')
        assert data.fetch_obs_demo(2).shape[0] == 2
        if torch.cuda.is_available():
            gpu_data = MotionPriorData(config, 'cuda')
            obs = gpu_data.fetch_obs_demo(2).reshape(2, 3, -1)
            reconstructed = gpu_data.convert_sample_to_frames(obs[0])
            assert reconstructed.is_cuda and torch.isfinite(reconstructed).all()
        print('PASS: direct motion loading and available CUDA data conversion')

    for name, module in list(sys.modules.items()):
        if name.startswith('smp_tools') and getattr(module, '__file__', None):
            assert Path(module.__file__).resolve().is_relative_to(PACKAGE_ROOT)
    print('PASS: all local dependencies are inside smp_tools')


if __name__ == '__main__':
    main()
