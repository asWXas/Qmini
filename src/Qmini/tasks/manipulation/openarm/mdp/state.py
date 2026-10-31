"""Read TCP state in the articulation's environment order."""


def tcp_pose_w(env):
    """Avoid a separate rigid-body view with different clone-prototype ordering."""
    robot = env.scene["robot"]
    body_id = robot.find_bodies("openarm_ee_tcp")[0][0]
    return robot.data.body_link_pos_w.torch[:, body_id], robot.data.body_link_quat_w.torch[:, body_id]
