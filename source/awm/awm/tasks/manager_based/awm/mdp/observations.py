# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def _goal_distance_xy(env, goal_distance: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_xy = asset.data.root_pos_w[:, :2]
    goal_xy = env.scene.env_origins[:, :2] + torch.tensor([goal_distance, 0.0], device=env.device)
    return torch.norm(goal_xy - root_xy, dim=-1)


def distance_to_goal(env, goal_distance: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _goal_distance_xy(env, goal_distance, asset_cfg).unsqueeze(-1)


def base_lin_vel_x(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 0:1]


def mean_wheel_speed(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.mean(vel, dim=1, keepdim=True)


def mean_leg_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.mean(pos, dim=1, keepdim=True)


def leg_actions(env, num_wheels: int) -> torch.Tensor:
    return env.action_manager.action[:, num_wheels:]


def wheel_contact_forces(env, sensor_name: str = "contact_forces", body_names: str | list[str] = "wheel_.*") -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    body_ids, _ = sensor.find_bodies(body_names, preserve_order=True)
    if len(body_ids) == 0:
        raise ValueError(f"No contact bodies matched {body_names}. Available: {sensor.body_names}")
    forces = sensor.data.net_forces_w[:, body_ids, :]
    return torch.norm(forces, dim=-1)
