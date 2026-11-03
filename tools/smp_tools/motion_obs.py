"""Motion-prior observations matching MimicKit, without simulator imports."""

import torch
import smp_tools.util.torch_util as torch_util


@torch.jit.script
def compute_tar_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, joint_rot, key_pos,
                    global_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    ref_root_pos = ref_root_pos.unsqueeze(-2)
    root_pos_obs = root_pos - ref_root_pos

    if (len(key_pos) > 0):
        key_pos = key_pos - root_pos.unsqueeze(-2)

    if (not global_obs):
        heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, root_pos.shape[1], 1))
        heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1],
                                                               heading_inv_rot_expand.shape[2]))
        root_pos_obs_flat = torch.reshape(root_pos_obs, [root_pos_obs.shape[0] * root_pos_obs.shape[1], root_pos_obs.shape[2]])
        root_pos_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_pos_obs_flat)
        root_pos_obs = torch.reshape(root_pos_obs_flat, root_pos.shape)

        root_rot = torch_util.quat_mul(heading_inv_rot_expand, root_rot)

        if (len(key_pos) > 0):
            heading_inv_rot_expand = heading_inv_rot_expand.unsqueeze(-2)
            heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, 1, key_pos.shape[2], 1))
            heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1] * heading_inv_rot_expand.shape[2],
                                                                   heading_inv_rot_expand.shape[3]))
            key_pos_flat = key_pos.reshape((key_pos.shape[0] * key_pos.shape[1] * key_pos.shape[2],
                                            key_pos.shape[3]))
            key_pos_flat = torch_util.quat_rotate(heading_inv_rot_flat, key_pos_flat)
            key_pos = key_pos_flat.reshape(key_pos.shape)

    if (root_height_obs):
        root_pos_obs[..., 2] = root_pos[..., 2]
    else:
        root_pos_obs = root_pos_obs[..., :2]

    root_rot_flat = torch.reshape(root_rot, [root_rot.shape[0] * root_rot.shape[1], root_rot.shape[2]])
    root_rot_obs_flat = torch_util.quat_to_tan_norm(root_rot_flat)
    root_rot_obs = torch.reshape(root_rot_obs_flat, [root_rot.shape[0], root_rot.shape[1], root_rot_obs_flat.shape[-1]])

    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1] * joint_rot.shape[2], joint_rot.shape[3]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1], joint_rot.shape[2] * joint_rot_obs_flat.shape[-1]])

    obs = [root_pos_obs, root_rot_obs, joint_rot_obs]
    if (len(key_pos) > 0):
        key_pos = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1], key_pos.shape[2] * key_pos.shape[3]])
        obs.append(key_pos)

    obs = torch.cat(obs, dim=-1)

    return obs

@torch.jit.script
def compute_disc_vel_obs(ref_root_rot, root_vel, root_ang_vel, dof_vel, global_obs, dof_vel_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor

    if (not global_obs):
        heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, root_vel.shape[1], 1))
        heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1],
                                                               heading_inv_rot_expand.shape[2]))

        root_vel_obs_flat = torch.reshape(root_vel, [root_vel.shape[0] * root_vel.shape[1], root_vel.shape[2]])
        root_vel_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_vel_obs_flat)
        root_vel_obs = torch.reshape(root_vel_obs_flat, root_vel.shape)

        root_ang_vel_obs_flat = torch.reshape(root_ang_vel, [root_ang_vel.shape[0] * root_ang_vel.shape[1], root_ang_vel.shape[2]])
        root_ang_vel_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_ang_vel_obs_flat)
        root_ang_vel_obs = torch.reshape(root_ang_vel_obs_flat, root_ang_vel.shape)
    else:
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel

    obs = [root_vel_obs, root_ang_vel_obs]

    if (dof_vel_obs):
        obs = obs + [dof_vel]

    obs = torch.cat(obs, dim=-1)

    return obs

@torch.jit.script
def compute_disc_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs, dof_vel_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool) -> Tensor

    pos_obs = compute_tar_obs(ref_root_pos=ref_root_pos,
                                               ref_root_rot=ref_root_rot,
                                               root_pos=root_pos,
                                               root_rot=root_rot,
                                               joint_rot=joint_rot,
                                               key_pos=key_pos,
                                               global_obs=global_obs,
                                               root_height_obs=root_height_obs)

    vel_obs = compute_disc_vel_obs(ref_root_rot=ref_root_rot,
                                  root_vel=root_vel,
                                  root_ang_vel=root_ang_vel,
                                  dof_vel=dof_vel,
                                  global_obs=global_obs,
                                  dof_vel_obs=dof_vel_obs)

    disc_obs = torch.cat([pos_obs, vel_obs], dim=-1)
    disc_obs = torch.reshape(disc_obs, [disc_obs.shape[0], -1])

    return disc_obs
