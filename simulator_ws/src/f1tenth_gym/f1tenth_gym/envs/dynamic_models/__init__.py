"""
This module contains the dynamic models available in the F1Tenth Gym.
Each submodule contains a single model, and the equations or their source is documented alongside it. Many of the models are from the CommonRoad repository, available here: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
"""

import warnings
from enum import Enum
import numpy as np

from .kinematic import vehicle_dynamics_ks, get_standardized_state_ks
from .single_track import vehicle_dynamics_st, get_standardized_state_st
from .multi_body import init_mb, vehicle_dynamics_mb, get_standardized_state_mb
from .utils import pid_steer, pid_accl
from .kinematic_mlp import get_standardized_state_kmlp
from .kinematic_discrete import get_standardized_state_kinematic
from typing import Optional

class DynamicModel(Enum):
    KS = 1  # Kinematic Single Track
    ST = 2  # Single Track
    MB = 3  # Multi-body Model
    KINEMATIC_MLP = 4
    KINEMATIC = 5
    KINEMATIC_NOSLIP_NOIMU_DIRECT_SPEED = 6
    DYNAMIC_MLP_RESIDUAL = 7
    SIMULATOR_GRU = 8
    DYNAMIC_SERVO_LAG = 9

    @staticmethod
    def from_string(model: str):
        if model == "ks":
            warnings.warn(
                "Chosen model is KS. This is different from previous versions of the gym."
            )
            return DynamicModel.KS
        elif model == "st":
            return DynamicModel.ST
        elif model == "mb":
            return DynamicModel.MB
        elif model == "kinematic_mlp":
            return DynamicModel.KINEMATIC_MLP
        elif model == "kinematic":
            return DynamicModel.KINEMATIC
        elif model == "kinematic_noslip_noimu_direct_speed":
            return DynamicModel.KINEMATIC_NOSLIP_NOIMU_DIRECT_SPEED
        elif model in ("dynamic_mlp_residual", "dynamic_mlp_residual_servo_lag",
                      "dynamic_mlp_residual_servo_lag_vx_delta_24d"):
            return DynamicModel.DYNAMIC_MLP_RESIDUAL
        elif model == "simulator_gru":
            return DynamicModel.SIMULATOR_GRU
        elif model in ("dynamic_servo_lag", "DYNAMIC_SERVO_LAG"):
            return DynamicModel.DYNAMIC_SERVO_LAG
        else:
            raise ValueError(f"Unknown model type {model}")

    def get_initial_state(self, pose=None, params: Optional[dict] = None):
        # Assert that if self is MB, params is not None
        if self == DynamicModel.MB and params is None:
            raise ValueError("MultiBody model requires parameters to be provided.")
        # initialize zero state
        if self == DynamicModel.KS:
            # state is [x, y, steer_angle, vel, yaw_angle]
            state = np.zeros(5)
        elif self == DynamicModel.ST:
            # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
            state = np.zeros(7)
        elif self == DynamicModel.MB:
            # state is a 29D vector
            state = np.zeros(29)
        elif self in (DynamicModel.KINEMATIC_MLP, DynamicModel.KINEMATIC,
                      DynamicModel.KINEMATIC_NOSLIP_NOIMU_DIRECT_SPEED,
                      DynamicModel.DYNAMIC_MLP_RESIDUAL,
                      DynamicModel.DYNAMIC_SERVO_LAG):
            # [x, y, steering, vx, yaw, yaw_rate, slip_angle, vy]
            state = np.zeros(8)
        elif self == DynamicModel.SIMULATOR_GRU:
            state = np.zeros(8)
        else:
            raise ValueError(f"Unknown model type {self}")

        # set initial pose if provided
        if pose is not None:
            state[0:2] = pose[0:2]
            state[4] = pose[2]

        # If state is MultiBody, we must inflate the state to 29D
        if self == DynamicModel.MB:
            state = init_mb(state, params)
        return state

    @property
    def f_dynamics(self):
        if self == DynamicModel.KS:
            return vehicle_dynamics_ks
        elif self == DynamicModel.ST:
            return vehicle_dynamics_st
        elif self == DynamicModel.MB:
            return vehicle_dynamics_mb
        elif self == DynamicModel.KINEMATIC_MLP:
            raise RuntimeError("KINEMATIC_MLP uses its discrete RaceCar update")
        elif self == DynamicModel.KINEMATIC:
            raise RuntimeError("KINEMATIC uses its discrete RaceCar update")
        elif self == DynamicModel.KINEMATIC_NOSLIP_NOIMU_DIRECT_SPEED:
            raise RuntimeError("KINEMATIC_NOSLIP_NOIMU_DIRECT_SPEED uses its discrete RaceCar update")
        elif self == DynamicModel.DYNAMIC_MLP_RESIDUAL:
            raise RuntimeError("DYNAMIC_MLP_RESIDUAL uses its discrete RaceCar update")
        elif self == DynamicModel.SIMULATOR_GRU:
            raise RuntimeError("SIMULATOR_GRU uses its discrete RaceCar update")
        elif self == DynamicModel.DYNAMIC_SERVO_LAG:
            raise RuntimeError("DYNAMIC_SERVO_LAG uses its discrete RaceCar update")
        else:
            raise ValueError(f"Unknown model type {self}")

    def get_standardized_state_fn(self):
        """
        This function returns the standardized state information for the model.
        This needs to be a function, because the state information is different for each model.
        Slip is not directly available from the MB model.
        """
        if self == DynamicModel.KS:
            return get_standardized_state_ks
        elif self == DynamicModel.ST:
            return get_standardized_state_st
        elif self == DynamicModel.MB:
            return get_standardized_state_mb
        elif self == DynamicModel.KINEMATIC_MLP:
            return get_standardized_state_kmlp
        elif self in (DynamicModel.KINEMATIC,
                      DynamicModel.KINEMATIC_NOSLIP_NOIMU_DIRECT_SPEED,
                      DynamicModel.DYNAMIC_MLP_RESIDUAL,
                      DynamicModel.DYNAMIC_SERVO_LAG):
            return get_standardized_state_kinematic
        elif self == DynamicModel.SIMULATOR_GRU:
            return get_standardized_state_kinematic
        else:
            raise ValueError(f"Unknown model type {self}")
