import pathlib
import copy
from typing import Dict
import torch

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask, downsample_mask, SequenceSampler
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer

# Based on omnid_image_dataset and pusht_dataset
class OmnidLowdimDataset(BaseLowdimDataset):
    def __init__(self,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        low_dim_obs_key = 'low_dim_obs',
        action_key = 'action',
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        dataset_path = pathlib.Path(dataset_path).absolute()
        zarr_path = dataset_path.joinpath(dataset_path.name + '.zarr')

        # Create replay buffer
        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path=zarr_path,
            keys=[low_dim_obs_key, action_key]
        )

        # Split data into validation/train
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        # Set up sequence sampler
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        # Store class members
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.low_dim_obs_key = low_dim_obs_key
        self.action_key = action_key
        self.val_mask = val_mask
        self.train_mask = train_mask
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
        )
        # Not sure why this happens...
        val_set.val_mask = ~self.val_mask
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):
        data = {
            'obs': sample[self.low_dim_obs_key],
            'action': sample[self.action_key],
        }
        return data
    
    def __getitem__(self, ndx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(ndx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

