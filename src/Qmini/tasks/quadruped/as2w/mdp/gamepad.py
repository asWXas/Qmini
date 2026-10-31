"""Linux Xbox joystick input, independent of the simulation viewer."""

import array
import fcntl
import os
import struct


class XboxGamepad:
    def __init__(self, path: str, deadzone: float):
        self._fd = None
        if not 0 <= deadzone < 1:
            raise ValueError("Gamepad deadzone must be in [0, 1)")
        self.deadzone = deadzone
        self._axes = {}
        try:
            try:
                self._fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot open Xbox controller {path}: {exc.strerror}. "
                    "Connect the controller or set env.commands.base_velocity.gamepad_path to its device path. "
                    "For normal playback, omit env.commands.base_velocity.xbox=true."
                ) from exc
            axis_map = array.array("B", [0] * 64)
            fcntl.ioctl(self._fd, 0x80406A32, axis_map)  # JSIOCGAXMAP
            self._axis_map = axis_map
            if not {0, 1, 3}.issubset(axis_map):
                raise RuntimeError("Controller must expose ABS_X, ABS_Y and ABS_RX axes")
        except Exception:
            self.close()
            raise

    def read(self):
        while True:
            try:
                event = os.read(self._fd, 8)
            except BlockingIOError:
                break
            except OSError as exc:
                self.close()
                raise RuntimeError("Xbox controller disconnected") from exc
            if len(event) != 8:
                self.close()
                raise RuntimeError("Xbox controller disconnected or returned an incomplete event")
            _, value, kind, index = struct.unpack("IhBB", event)
            if kind & 0x02:
                self._axes[self._axis_map[index]] = max(-1.0, value / 32767.0)

        def axis(code):
            value = -self._axes.get(code, 0.0)
            magnitude = max(abs(value) - self.deadzone, 0.0) / (1.0 - self.deadzone)
            return magnitude if value >= 0 else -magnitude

        return axis(1), axis(0), axis(3)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        self.close()
