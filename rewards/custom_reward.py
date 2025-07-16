from __future__ import annotations
"""
Custom reward function optimized for PPO and DQN agents in CARLA.

💡 **July 2025 hot‑fix** – Agents were learning to creep along at
1‑2 m/s.  The speed shaping has been rewritten so that *very* low
speed now produces a **large negative reward**, while hitting the
configured target speed becomes the strongest positive incentive.
The per‑algorithm weight table has been retuned accordingly.

Key design choices
------------------
* **Smooth, differentiable components** – avoids the "flat spots" that can
  freeze learning.
* **Algorithm‑specific weight & clip tables** – prevents value‑function
  explosion in DQN while letting PPO exploit a slightly broader range.
* **Linear speed mapping** – ``speed_ratio → 2·ratio − 1`` →
  −1 at 0 m/s, +1 at target‑speed.
* **Minimal external API change** – call signature of ``calculate_reward``
  is identical to the original implementation; pass ``algorithm='DQN'`` or
  ``'PPO'`` via the ``config`` dict to switch scaling.
"""

from typing import Dict, Any, Tuple

import numpy as np
import carla  # type: ignore

from .base_reward import BaseReward


class CustomReward(BaseReward):
    """Reward shaped for stable PPO/DQN training in CARLA."""

    # ------------------------------------------------------------------
    #  Weight & clipping table – retuned July 2025
    # ------------------------------------------------------------------
    _ALG_TABLE = {
        "PPO": dict(angle=0.25, distance=0.6, speed=0.6, collision=1.0,
                     clip_low=-1.5, clip_high=2.0),
        "DQN": dict(angle=0.15, distance=0.6, speed=0.4, collision=1.0,
                     clip_low=-1.0, clip_high=1.5),
    }

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    def __init__(self, config: Dict[str, Any] | None = None):
        """Create a new :class:`CustomReward`.

        Parameters
        ----------
        config
            Optional dictionary overriding any of the default parameters.
            Notable keys::

                algorithm: "PPO" | "DQN" (default "PPO")
                target_speed: float, m/s (default 6.0)
                lane_width: float, m (default 4.0)
        """
        default_config: Dict[str, Any] = dict(
            algorithm="PPO",  # or "DQN"
            target_speed=6.0,
            lane_width=4.0,
        )
        if config:
            default_config.update(config)
        super().__init__(default_config)

        algo = str(self.config["algorithm"]).upper()
        if algo not in self._ALG_TABLE:
            raise ValueError(f"Unsupported algorithm '{algo}'. Use 'PPO' or 'DQN'.")
        self.weights = self._ALG_TABLE[algo]
        self.reset()

    # ------------------------------------------------------------------
    # House‑keeping helpers
    # ------------------------------------------------------------------

    def reset(self):
        self.total_distance: float = 0.0
        self.last_location: carla.Location | None = None

    # ------------------------------------------------------------------
    #  Utility math – each returns value ∈ [‑1, 1]
    # ------------------------------------------------------------------
    @staticmethod
    def _smooth_cos(angle_deg: float) -> float:
        return float(np.cos(np.radians(angle_deg)))

    def _exp_distance(self, dist_m: float) -> float:
        lane_width = float(self.config["lane_width"])
        return float(-np.tanh(dist_m / (lane_width / 2)))

    def _speed_term(self, speed: float) -> float:
        """
        Piece-wise linear mapping, always in [-1, 1]:

        *  0 m/s → -1             (strong penalty for parking)
        *  target_speed → +1     (ideal cruising)
        *  >2×target_speed → 0   (discourage speeding without punishing hard)
        """
        tgt = float(self.config["target_speed"])

        if speed <= tgt:                       # [0, tgt]  ⇒  [-1, +1]
            return (speed / tgt) * 2.0 - 1.0
        else:                                  # (tgt, 2·tgt] ⇒  (1, 0]
            overshoot = min(speed - tgt, tgt)  # cap after 2×tgt
            return 1.0 - overshoot / tgt

    @staticmethod
    def _collision_penalty(detected: bool) -> float:
        return -1.0 if detected else 0.0

    # ------------------------------------------------------------------
    #  World helpers (unchanged from the original where possible)
    # ------------------------------------------------------------------
    def _get_road_orientation(self, vehicle: carla.Vehicle) -> float:
        waypoint = vehicle.get_world().get_map().get_waypoint(vehicle.get_location())
        yaw = waypoint.transform.rotation.yaw if waypoint else vehicle.get_transform().rotation.yaw
        return (yaw + 180) % 360 - 180  # → [‑180, 180]

    def _get_distance_from_center(self, vehicle: carla.Vehicle) -> Tuple[float, bool]:
        waypoint = vehicle.get_world().get_map().get_waypoint(vehicle.get_location())
        if waypoint is None:
            return 999.0, False
        dist = vehicle.get_location().distance(waypoint.transform.location)
        return dist, True

    # ------------------------------------------------------------------
    #  Main reward function
    # ------------------------------------------------------------------
    def calculate_reward(
        self,
        vehicle: carla.Vehicle,
        episode_step: int,
        collision_detected: bool,
        **kwargs,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        # ---------- progress term ----------
        if self.last_location is None:
            delta_s = 0.0
        else:
            disp      = vehicle.get_location() - self.last_location
            forward_y = np.radians(self._get_road_orientation(vehicle))
            forward_v = np.array([np.cos(forward_y), np.sin(forward_y), 0])
            delta_s   = float(np.dot(np.array([disp.x, disp.y, disp.z]), forward_v))
        r_progress = np.tanh(delta_s / (self.config["target_speed"] * 0.05))

        # --------------------------------------------------------------
        # 1.  Angle alignment term
        # --------------------------------------------------------------
        theta_v = (vehicle.get_transform().rotation.yaw + 180) % 360 - 180
        theta_r = self._get_road_orientation(vehicle)
        diff = abs(theta_v - theta_r)
        diff = diff if diff <= 180 else 360 - diff
        r_angle = self._smooth_cos(diff)  # ∈ [‑1, 1]

        # --------------------------------------------------------------
        # 2.  Lateral distance term
        # --------------------------------------------------------------
        dist, on_road = self._get_distance_from_center(vehicle)
        r_dist = self._exp_distance(dist) if on_road else -1.0

        # --------------------------------------------------------------
        # 3.  Speed term
        # --------------------------------------------------------------
        vel = vehicle.get_velocity()
        speed = float(np.hypot(vel.x, vel.y))
        r_speed = self._speed_term(speed)

        # --------------------------------------------------------------
        # 4.  Collision term
        # --------------------------------------------------------------
        r_coll = self._collision_penalty(collision_detected)

        # --------------------------------------------------------------
        # Weighted sum + clipping
        # --------------------------------------------------------------
        w = self.weights
        reward = (
            0.8 * r_progress +          # new
            w["angle"] * r_angle
            + w["distance"] * r_dist
            + w["speed"] * r_speed
            + w["collision"] * r_coll
        )
        reward = float(np.clip(reward, w["clip_low"], w["clip_high"]))

        # --------------------------------------------------------------
        # Book‑keeping & diagnostics
        # --------------------------------------------------------------
        if self.last_location is not None:
            self.total_distance += vehicle.get_location().distance(self.last_location)
        self.last_location = vehicle.get_location()

        done = bool(collision_detected)
        info = dict(
            angle=r_angle,
            distance=r_dist,
            speed=r_speed,
            collision=r_coll,
            raw_reward=reward,
            total_distance=self.total_distance,
            distance_from_center=dist,
            is_on_road=on_road,
            speed_mps=speed,
            episode_step=episode_step,
            progress=r_progress,
        )
        return reward, done, info

    # ------------------------------------------------------------------
    #  Optional helper for external inspection
    # ------------------------------------------------------------------
    def get_info(self) -> Dict[str, Any]:  # pragma: no cover
        return dict(config=self.config)
