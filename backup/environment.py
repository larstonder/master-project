"""Python script to render images using the OmniRe model.
This script is designed to work with the DriveStudio framework and
is tailored for the NuPlan dataset.
It handles the rendering of images based on the simulation state
and the camera parameters.
"""

import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R
from state_types import State, EnvState
from setup import OmniReSetup
from drivestudio.datasets.base.pixel_source import get_rays


OPENCV2DATASET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

class OmniRe:
    """
    OmniReModel class for rendering images using the OmniRe model.
    This class is designed to work with the DriveStudio framework and
    is tailored for the NuPlan dataset.
    It handles the rendering of images based on the simulation state
    and the camera parameters.
    """

    def __init__(self, setup: OmniReSetup):
        self.data_cfg = setup.data_cfg
        self.train_cfg = setup.train_cfg
        self.trainer = setup.trainer
        self.dataset = setup.dataset
        self.device = setup.device
        self.camera_matrix_cache = {}

    def get_sensor_output(self, state: State) -> EnvState:
        """
        Generate sensor output (RGB image) for the given simulation state.

        Args:
            state (State): The current simulation state.

        Returns:
            dict: Sensor outputs including rendered RGB image and depth
        """
        # 1. Get ego position and heading
        pos = state.ego_pos
        position = torch.tensor([pos.x, pos.y, pos.z], dtype=torch.float32, device=self.device)
        heading = float(pos.heading)
        
        # 2. Convert heading (yaw) to quaternion [w, x, y, z]
        quat = R.from_euler('z', heading).as_quat()  # [x, y, z, w]
        quat = np.roll(quat, 1)  # to [w, x, y, z]
        
        rotation = torch.tensor(quat, dtype=torch.float32, device=self.device)
        # 3. Compute camera-to-world matrix (c2w)
        # Following DriveStudio convention: only yaw rotation
        c2w = torch.eye(4, dtype=torch.float32, device=self.device)
        rot_mat = torch.from_numpy(R.from_quat(quat).as_matrix()).to(self.device)
        c2w[:3, :3] = rot_mat
        c2w[:3, 3] = position
        
        # 4. Get intrinsics and image size from dataset (use front camera, id=0)
        cam_id = 0
        cam_data = self.dataset.pixel_source.camera_data[cam_id]
        intrinsics = cam_data.intrinsics[0].to(self.device)
        H, W = cam_data.HEIGHT, cam_data.WIDTH
        
        # 5. Generate per-pixel rays
        x, y = torch.meshgrid(torch.arange(W, device=self.device), torch.arange(H, device=self.device), indexing='xy')
        # x, y = x.flatten(), y.flatten()
        origins, viewdirs, direction_norm = get_rays(x, y, c2w, intrinsics)
        origins = origins.reshape(H, W, 3)
        viewdirs = viewdirs.reshape(H, W, 3)
        direction_norm = direction_norm.reshape(H, W, 1)

        # 6. Normalized time (between 0 and 1)
        # Use the closest frame in the dataset for normalization
        normed_time = 0.0
        if hasattr(self.dataset.pixel_source, 'normalized_time') and self.dataset.pixel_source.normalized_time is not None:
            # Convert state.timestamp to microseconds if it's a TimePoint
            ts = state.timestamp
            if hasattr(ts, 'time_us'):
                ts_val = float(ts.time_us)
            elif hasattr(ts, 'time_s'):
                ts_val = float(ts.time_s * 1e6)
            else:
                ts_val = float(ts)
            times = self.dataset.pixel_source.normalized_time
            normed_time = float(times[self.dataset.pixel_source.find_closest_timestep(ts_val)])

        # 7. Build cam_infos and image_infos dicts
        cam_infos = {
            "camera_to_world": c2w,
            "intrinsics": intrinsics,
            "height": torch.tensor(H, dtype=torch.long, device=self.device),
            "width": torch.tensor(W, dtype=torch.long, device=self.device),
            "cam_name": cam_data.cam_name,
            "cam_id": torch.tensor(cam_id, dtype=torch.long, device=self.device),
        }
        image_infos = {
            "origins": origins,
            "viewdirs": viewdirs,
            "direction_norm": direction_norm,
            "pixel_coords": torch.stack([y.float() / H, x.float() / W], dim=-1).reshape(H, W, 2),
            "normed_time": torch.full((H, W), normed_time, dtype=torch.float32, device=self.device),
            "img_idx": torch.full((H, W), 0, dtype=torch.long, device=self.device),
            "frame_idx": torch.full((H, W), 0, dtype=torch.long, device=self.device),
        }
        
        # 8. Render using the trainer (novel_view=True)
        with torch.no_grad():
            outputs = self.trainer(image_infos=image_infos, camera_infos=cam_infos, novel_view=True)
        
        # 9. Return sensor outputs (clip rgb to [0,1])
        rgb = outputs["rgb"].clamp(0., 1.).cpu().numpy()
        depth = outputs["depth"].cpu().numpy() if "depth" in outputs else None
        sensor_output = EnvState(rgb, depth)
        return sensor_output
