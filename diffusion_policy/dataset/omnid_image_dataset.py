import pathlib

from diffusion_policy.dataset.base_dataset import BaseImageDataset

class OmnidImageDataset(BaseImageDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        # TODO remove and develop further
        x = 1/0