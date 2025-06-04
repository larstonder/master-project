"""Python script to render images using the OmniRe model.
This script is designed to work with the DriveStudio framework and
is tailored for the NuPlan dataset.
It handles the rendering of images based on the simulation state
and the camera parameters.
"""

import os
import torch
import numpy as np
import torch
import cv2
import copy
from abc import ABC, abstractmethod
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict
from scipy.spatial.transform import Rotation as R
from monarch.typings.state_types import SystemState, EnvState, VehicleState
from monarch.utils.path_utils import local_import_context, use_path
from monarch.utils.misc import import_str  # pylint: disable=import-error
from monarch.rendering.renderer import Renderer

with local_import_context("./drivestudio", True):
    from datasets.driving_dataset import DrivingDataset
    from datasets.base.pixel_source import get_rays
    from models.trainers.scene_graph import MultiTrainer

CORRECTION_MATRIX = np.array([[-0.90322894,  0.42915905,  0.        ],
                              [-0.42915905, -0.90322894,  0.        ],
                              [ 0.,          0.,          1.        ]])

class OmniRe(Renderer):
    """
    OmniReModel class for rendering images using the OmniRe model.
    This class is designed to work with the DriveStudio framework and
    is tailored for the NuPlan dataset.
    It handles the rendering of images based on the simulation state
    and the camera parameters.
    """
        
    def __init__(
        self,
        data_path: str,
        checkpoint_path: str,
        cam_id: int = 0,
        start_timestep: int = 0,
    ):
        print(f"Initializing OmniRe renderer with data path: {data_path} and checkpoint path: {checkpoint_path}")
        self.data_path = Path(data_path)
        self.raw_checkpoint_path = Path(checkpoint_path)
        self.cam_id = cam_id
        self.start_timestep = start_timestep
        self.initialize()
        print("OmniRe renderer initialized successfully")
    
    def initialize(self):
        self.checkpoint_path = self.raw_checkpoint_path / "config.yaml"
        self.cfg = OmegaConf.load(self.checkpoint_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = DrivingDataset(data_cfg=self.cfg.data)
        
        self.reset()
        self._initialize_frame_data()

    def reset(self):
        print(f"Initializing trainer {self.cfg.trainer.type}")
        
        self.trainer = MultiTrainer(
            **self.cfg.trainer,
            num_timesteps=self.dataset.num_img_timesteps,
            model_config=self.cfg.model,
            num_train_images=len(self.dataset.train_image_set),
            num_full_images=len(self.dataset.full_image_set),
            test_set_indices=self.dataset.test_timesteps,
            scene_aabb=self.dataset.get_aabb().reshape(2, 3),
            device=self.device,
        )
        
        ckpt_path = self.raw_checkpoint_path / "checkpoint_final.pth"

        # Load checkpoint
        self.trainer.resume_from_checkpoint(ckpt_path=ckpt_path, load_only_model=True)

        print(
            f"Resuming from checkpoint: {ckpt_path}, starting at step {self.trainer.step}"
        )

    def _initialize_frame_data(self):
        """Initialize the dataset and frame data"""
        render_traj = self.dataset.get_novel_render_traj()
        self.frame_data = self.dataset.pixel_source.prepare_novel_view_render_data(
            self.dataset.pixel_source, render_traj["front_center_interp"]
        )[0]
    
    @property
    def name(self) -> str:
        return "OmniRe"

    def _get_delta_pos(self, last_pos: VehicleState, current_pos: VehicleState) -> VehicleState:
        """Get the delta position between two positions"""
        delta_x = float(current_pos.x) - float(last_pos.x)
        delta_y = float(current_pos.y) - float(last_pos.y)
        delta_z = float(current_pos.z) - float(last_pos.z)
        delta_heading = float(current_pos.heading) - float(last_pos.heading)
        return VehicleState(x=delta_x, y=delta_y, z=delta_z, heading=delta_heading, id=current_pos.id)
    
    def _compute_vehicle_deltas(self, last_state: SystemState, current_state: SystemState) -> list[VehicleState]:
        vehicle_deltas = []
        if hasattr(current_state, 'vehicle_pos_list') and hasattr(last_state, 'vehicle_pos_list'):
            for i in range(len(current_state.vehicle_pos_list)):
                
                current_id_to_check = current_state.vehicle_pos_list[i].id
                for j in range(len(last_state.vehicle_pos_list)):
                    if last_state.vehicle_pos_list[j].id == current_id_to_check:
                        delta_pos = self._get_delta_pos(last_state.vehicle_pos_list[j], current_state.vehicle_pos_list[i])                    
                        vehicle_deltas.append(delta_pos)
                        break
        return vehicle_deltas
    
    def _edit_nodes(self, vehicle_deltas: list[VehicleState] = None, apply_correction: bool = True):
        """Edit the nodes in the scene graph based on vehicle position deltas.
        
        Args:
            vehicle_deltas: List of VehicleState objects representing the change in position and heading for each vehicle.
        """
        try:
            if vehicle_deltas is None or len(vehicle_deltas) == 0:
                print("No vehicle deltas provided, skipping node editing")
                return
            
            if "RigidNodes" in self.trainer.gaussian_classes.keys():
                rigid_model = self.trainer.models["RigidNodes"]
                
                with torch.no_grad():
                    for vehicle_state in vehicle_deltas:
                        instance_id = vehicle_state.id
                        
                        if instance_id is None:
                            print("Error: Vehicle delta id is None. Skipping edit for this vehicle.")
                            continue
                        
                        if hasattr(rigid_model, 'track_token_to_model_id_map') and instance_id in rigid_model.track_token_to_model_id_map:
                            model_id = rigid_model.track_token_to_model_id_map[instance_id]
                        else:
                            try:
                                dataset_id = int(instance_id)
                                if hasattr(rigid_model, 'dataset_id_to_model_id_map') and dataset_id in rigid_model.dataset_id_to_model_id_map:
                                    model_id = rigid_model.dataset_id_to_model_id_map[dataset_id]
                                else:
                                    continue
                            except ValueError:
                                continue
                                                
                        delta_vector = np.array([vehicle_state.x, vehicle_state.y, vehicle_state.z])
                        
                        if apply_correction:
                            correction_matrix = torch.tensor(CORRECTION_MATRIX, device=rigid_model.device)
                            delta_vector_tensor = torch.tensor(delta_vector, device=rigid_model.device)
                            updated_vector = (correction_matrix @ delta_vector_tensor).float()
                            
                            delta_vector = updated_vector
                        
                        delta_x = delta_vector[0]
                        delta_y = delta_vector[1]
                        delta_z = delta_vector[2]
                        
                        delta_translation = torch.tensor([delta_y, delta_x, delta_z], device=rigid_model.device).float()
                        delta_rotation_quat = torch.tensor([1.0, 0.0, 0.0, vehicle_state.heading], device=rigid_model.device)  # Convert heading to quaternion as needed
                        
                        rigid_model.translate_instance(model_id, delta_translation)
                        rigid_model.rotate_instance(model_id, delta_rotation_quat)
                
        except Exception as e:
            print(f"Error in edit_nodes: {str(e)}")
            traceback.print_exc()

    def _update_camera_pose(self, frame_data: dict, original_state: SystemState, current_state: SystemState, apply_correction: bool = True):
        c2w = frame_data["cam_infos"]["camera_to_world"].to(self.device)
        
        ego_delta = self._get_delta_pos(original_state.ego_pos, current_state.ego_pos)
        
        delta_x, delta_y, delta_z = ego_delta.x, ego_delta.y, ego_delta.z
        
        if apply_correction:
            delta_vector = np.array([ego_delta.x, ego_delta.y, ego_delta.z])
            correction_matrix = CORRECTION_MATRIX.astype(delta_vector.dtype)
            updated_vector = correction_matrix @ delta_vector
            
            delta_x = updated_vector[0]
            delta_y = updated_vector[1]
            delta_z = updated_vector[2]
        
        delta_translation = torch.tensor([delta_y, delta_x, delta_z], device=self.device)
        
        c2w[:3, 3] = c2w[:3, 3] + delta_translation
        
        current_rotation = c2w[:3, :3].cpu().numpy()
        
        current_rot_obj = R.from_matrix(current_rotation)
        delta_heading_rot = R.from_euler('y', ego_delta.heading, degrees=False)
        
        new_rotation = current_rot_obj * delta_heading_rot
        c2w[:3, :3] = torch.tensor(new_rotation.as_matrix(), device=self.device)

        frame_data["cam_infos"]["camera_to_world"] = c2w
    
    def _prepare_rays(self, frame_data: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cam_id = 0
        
        c2w = frame_data["cam_infos"]["camera_to_world"]
        
        H, W = (
            self.dataset.pixel_source.camera_data[cam_id].HEIGHT,
            self.dataset.pixel_source.camera_data[cam_id].WIDTH,
        )

        x, y = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing="xy",
        )
        x, y = x.flatten(), y.flatten()

        intrinsics = frame_data["cam_infos"]["intrinsics"]

        x, y, c2w, intrinsics = (
            x.to(self.device),
            y.to(self.device),
            c2w.to(self.device),
            intrinsics.to(self.device),
        )

        origins, viewdirs, direction_norm = get_rays(x, y, c2w, intrinsics)
        origins = origins.reshape(H, W, 3)
        viewdirs = viewdirs.reshape(H, W, 3)
        direction_norm = direction_norm.reshape(H, W, 1)

        frame_data["image_infos"]["origins"] = origins
        frame_data["image_infos"]["viewdirs"] = viewdirs
        frame_data["image_infos"]["direction_norm"] = direction_norm
        frame_data["cam_infos"]["height"] = torch.tensor(H, device=self.device)
        frame_data["cam_infos"]["width"] = torch.tensor(W, device=self.device)
    
    def _render_and_extract_rgb(self, frame_data: dict) -> EnvState:
        with torch.no_grad():
            # Move data to GPU
            for key, value in frame_data["cam_infos"].items():
                frame_data["cam_infos"][key] = value.cuda(non_blocking=True)
            for key, value in frame_data["image_infos"].items():
                frame_data["image_infos"][key] = value.cuda(non_blocking=True)

            # Perform rendering
            outputs = self.trainer(
                image_infos=frame_data["image_infos"],
                camera_infos=frame_data["cam_infos"],
                novel_view=True,
            )

            # Extract RGB image and mask
            rgb = outputs["rgb"].cpu().numpy().clip(min=1.0e-6, max=1 - 1.0e-6)

            return EnvState(rgb_image=rgb, depth=None)
        

    def get_sensor_input(self, original_state: SystemState, last_state: SystemState, current_state: SystemState, apply_correction: bool = True) -> EnvState:
        frame_data = copy.deepcopy(self.frame_data)
        self.trainer.set_eval()
        
        # ---- UPDATE OTHER VEHICLES ----
        vehicle_deltas = self._compute_vehicle_deltas(last_state, current_state)
        ego_delta = self._get_delta_pos(original_state.ego_pos, current_state.ego_pos)
    
        self._edit_nodes(vehicle_deltas, apply_correction)
        self._update_camera_pose(frame_data, original_state, current_state, apply_correction)
        self._prepare_rays(frame_data)

        return self._render_and_extract_rgb(frame_data)
