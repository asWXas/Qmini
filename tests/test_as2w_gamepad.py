import struct
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from Qmini.tasks.quadruped.as2w.mdp.commands import GroupedVelocityCommand
from Qmini.tasks.quadruped.as2w.mdp.gamepad import XboxGamepad


def test_joystick_mapping_deadzone_and_disconnect():
    pad = XboxGamepad.__new__(XboxGamepad)
    pad._fd = 123
    pad.deadzone = 0.1
    pad._axes = {}
    pad._axis_map = [0, 1, 2, 3]
    events = [struct.pack("IhBB", 0, v, 0x82, i) for i, v in ((0, 100), (1, -32768), (3, 32767))]
    with patch("os.read", side_effect=[*events, BlockingIOError()]):
        assert pad.read() == (1.0, 0.0, -1.0)
    with patch("os.read", side_effect=OSError()), patch("os.close") as close:
        try:
            pad.read()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Disconnect must stop playback")
        close.assert_called_once_with(123)


def test_gamepad_overrides_heading_and_resampling():
    term = GroupedVelocityCommand.__new__(GroupedVelocityCommand)
    term.cfg = SimpleNamespace(xbox=True, ranges=SimpleNamespace(lin_vel_x=(-2, 1), lin_vel_y=(-1, 1), ang_vel_z=(-1, 1)))
    term.vel_command_b = torch.zeros(2, 3)
    term.is_standing_env = torch.ones(2, dtype=torch.bool)
    term.is_heading_env = torch.ones(2, dtype=torch.bool)
    term.command_type = torch.zeros(2, dtype=torch.long)
    term._gamepad = Mock()
    term._gamepad.read.return_value = (-0.5, 0.25, -0.75)
    term._resample_command([0, 1])
    term._update_command()
    torch.testing.assert_close(term.vel_command_b, torch.tensor([[-1, 0.25, -0.75]]).repeat(2, 1))
    assert not term.is_heading_env.any()
    assert not term.is_standing_env.any()
    term._gamepad.read.return_value = (0, 0, 0)
    term._update_command()
    assert term.is_standing_env.all()
    assert not term.vel_command_b.any()
