from typing import Dict

import numpy as np
import pathlib
import copy
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask, downsample_mask, SequenceSampler
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


# Based on real_pusht_image_dataset and pusht_image_dataset
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
        dataset_path = pathlib.Path(dataset_path).absolute()
        zarr_path = dataset_path.joinpath(dataset_path.name + '.zarr')

        rgb_keys = list()
        low_dim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                low_dim_keys.append(key)
        obs_keys = rgb_keys + low_dim_keys

        # only take first k obs?
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in obs_keys:
                key_first_k[key] = n_obs_steps
        
        # Create replay buffer
        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path=zarr_path,
            keys=obs_keys + ['action']
        )

        # Split data into validation/train
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
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
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k
        )

        # Store class members
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.obs_keys = obs_keys
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

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Action normalizer
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer['action'])

        # Low dim observations
        for key in self.low_dim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer[key])

        # Images
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])
    
    def __len__(self):
        return len(self.sampler)
    
    def __getitem__(self, ndx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(ndx)

        # This comment from real_pusht_image_dataset
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rbg_keys:
            # Move channel last to channel first
            # T,H,W,C -> T,C,H,W
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.

            # save RAM
            del data[key]

        for key in self.low_dim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)

            # save RAM
            del data[key]

        action = data['action'].astype(np.float32)

        # handle latency by dropping forst n_latency_steps actions
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
        
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
        }
        return torch_data