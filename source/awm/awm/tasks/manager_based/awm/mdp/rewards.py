# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Computes 2D (x-y plane) distance from the robot base to a fixed goal point
# located at +goal_distance along each environment's x-axis origin.
def _goal_distance_xy(env, goal_distance: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_xy = asset.data.root_pos_w[:, :2]
    goal_xy = env.scene.env_origins[:, :2] + torch.tensor([goal_distance, 0.0], device=env.device)
    return torch.norm(goal_xy - root_xy, dim=-1)


# Dense progress reward term:
# keeps track of previous goal distance and rewards reduction in distance
# at each step (positive when moving toward the goal, negative otherwise).
class progress_to_goal(ManagerTermBase):
    """Reward for making progress toward a fixed goal in +x direction."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.prev_dist = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        asset_cfg = self.cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        goal_distance = self.cfg.params["goal_distance"]
        dist = _goal_distance_xy(self._env, goal_distance, asset_cfg)
        self.prev_dist[env_ids] = dist[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        goal_distance: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        dist = _goal_distance_xy(env, goal_distance, asset_cfg)
        progress = self.prev_dist - dist
        self.prev_dist[:] = dist
        return progress

# Sparse success reward:
# returns 1.0 when robot is inside goal_radius from the target, else 0.0.
def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    goal_distance: float,
    goal_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    dist = _goal_distance_xy(env, goal_distance, asset_cfg)
    return (dist < goal_radius).float()
