"""
Setup functions for models

"""

import os
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import torch
from path_utils import local_import_context

with local_import_context("drivestudio"):
    from utils.misc import import_str  # pylint: disable=import-error
    from datasets.driving_dataset import DrivingDataset  # pylint: disable=import-error
    from models.trainers.scene_graph import MultiTrainer

OPENCV2DATASET = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)


# ---------------------------------------ENVIRONMENT---------------------------------------
class OmniReSetup:
    """
    OmniReSetup class that initializes the environment model with a trained neural renderer.
    It loads the configuration and checkpoint files, sets up the device,
    and prepares the dataset for rendering.
    """

    def __init__(self, data_path: str, checkpoint_path: str, cam_id: int = 0, start_timestep: int = 0):
        self.data_path = Path(data_path)
        print(f"Loading data from: {data_path}")
        
        self.cam_id = cam_id
        self.start_timestep = start_timestep
        
        self.cam_to_ego = self._load_and_transform_extrinsics()
        self.world_to_ego = self._calculate_world_anchor_transform()
        
        print(f"Data loaded.")
        
        raw_checkpoint_path = Path(checkpoint_path)
        checkpoint_path = raw_checkpoint_path / "config.yaml"

        print(f"Loading checkpoint from: {raw_checkpoint_path}")

        # Load configuration
        # self.data_cfg = OmegaConf.load(config_path)
        self.cfg = OmegaConf.load(checkpoint_path)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dataset = DrivingDataset(data_cfg=self.cfg.data)
        
        print(f"Initializing trainer {self.cfg.trainer.type}")

        # Setup trainer
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
        
        # Evaluate that the trainer is initialized correctly
        if self.trainer is None:
            raise ValueError("Trainer is not initialized correctly")
        
        print(f"Trainer type: {type(self.trainer)}")
        
        print("TRAINER INIT DONE")

        ckpt_path = raw_checkpoint_path / "checkpoint_final.pth"

        # Load checkpoint
        self.trainer.resume_from_checkpoint(ckpt_path=ckpt_path, load_only_model=True)
        
        print(f"Resuming from checkpoint: {ckpt_path}, starting at step {self.trainer.step}")
    
    def _load_and_transform_extrinsics(self) -> np.ndarray:
        """Loads camera extrinsics and applies OPENCV2DATASET transformation."""
        raw_cam_to_ego_path = os.path.join(self.data_path, "extrinsics", f"{self.cam_id}.txt")
        
        if not os.path.exists(raw_cam_to_ego_path):
            raise FileNotFoundError(f"Extrinsics file not found: {raw_cam_to_ego_path}")
        
        raw_extrinsics = np.loadtxt(raw_cam_to_ego_path)
        cam_to_ego = raw_extrinsics @ OPENCV2DATASET
        return cam_to_ego
    
    def _calculate_world_anchor_transform(self) -> np.ndarray:
        """Calculates the transformation from nuPlan sim world to OmniRe world."""
        ego_pose_start_path = os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        if not os.path.exists(ego_pose_start_path):
            raise FileNotFoundError(f"OmniRe reference ego_pose file not found: {ego_pose_start_path}")
        ego_to_world_start = np.loadtxt(ego_pose_start_path)
        world_to_ego = np.linalg.inv(ego_to_world_start)
        return world_to_ego
