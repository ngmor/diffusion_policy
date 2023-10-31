from typing import Any
from enum import Enum
from enum import auto as enum_auto
import pathlib
import numpy as np

import hydra
from omegaconf import OmegaConf

import sensor_msgs.msg
import geometry_msgs.msg

from diffusion_policy.common.replay_buffer import ReplayBuffer

from omnid_bag import decimate

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

PACKAGE_PATH = pathlib.Path(__file__).parent.parent.parent

class DataType(Enum):
    TIMESTEP = enum_auto()
    ACTION = enum_auto()
    LOW_DIM_OBS = enum_auto()
    IMG = enum_auto()

class DataLocator:
    def __init__(self, dtype: DataType, ndx: int):
        self._dtype = dtype
        self._ndx = ndx
    def __repr__(self) -> str:
        return f'({self.dtype.name}, {self.ndx})'
    @property
    def dtype(self) -> DataType:
        return self._dtype
    @property
    def ndx(self) -> int:
        return self._ndx

def place_scalar_attribute(obj: Any, attr: str, namespace: str, locator_dict: dict, low_dim_obs_list: list, actions_list: list):
    if hasattr(obj, attr):
        full_name = namespace + '.' + attr

        if getattr(obj, attr) == 'low_dim':
            dtype = DataType.LOW_DIM_OBS
            out_list = low_dim_obs_list
        elif getattr(obj, attr) == 'action':
            dtype = DataType.ACTION
            out_list = actions_list
        else:
            raise Exception(f'Unexpected value for data type: {full_name}. Should be \'low_dim\' or \'action\'')
        
        locator_dict[attr] = DataLocator(dtype, len(out_list))
        out_list.append(full_name)

# TODO make this recursive?
def place_sub_attributes(obj: Any, attr: str, sub_attrs: list, namespace: str, locator_dict: dict, low_dim_obs_list: list, actions_list: list):
    if hasattr(obj, attr):
        full_name = namespace + '.' + attr

        temp_dict = {}

        for sub_attr in sub_attrs:
            place_scalar_attribute(getattr(obj, attr), sub_attr, full_name, temp_dict, low_dim_obs_list, actions_list)

        if len(temp_dict.keys()) > 0:
            locator_dict[attr] = temp_dict

@hydra.main(
    version_base=None,
    config_path=str(PACKAGE_PATH.joinpath('diffusion_policy', 'config', 'bags'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    # Init replay buffer for handling output
    output_dir = pathlib.Path(cfg.output_path)
    assert output_dir.parent.is_dir()
    zarr_path = str(output_dir.joinpath(output_dir.name + '.zarr').absolute())
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode='w')

    # Determine what data will go where in the inputs to the model
    topics = []
    low_dim_obs_names = []
    actions_names = []

    # Determine where to place joint state data according to configuration
    joint_states = {}
    if hasattr(cfg, 'joint_states'):
        for topic in cfg.joint_states:
            topics.append(topic.topic)

            joint_states[topic.topic] = {}

            for joint in topic.joints:
                joint_states[topic.topic][joint.name] = {}

                namespace = topic.topic + '.' + joint.name

                place_scalar_attribute(
                    obj=joint,
                    attr='position',
                    namespace=namespace,
                    locator_dict=joint_states[topic.topic][joint.name],
                    low_dim_obs_list=low_dim_obs_names,
                    actions_list=actions_names
                )
                place_scalar_attribute(
                    obj=joint,
                    attr='velocity',
                    namespace=namespace,
                    locator_dict=joint_states[topic.topic][joint.name],
                    low_dim_obs_list=low_dim_obs_names,
                    actions_list=actions_names
                )
                place_scalar_attribute(
                    obj=joint,
                    attr='effort',
                    namespace=namespace,
                    locator_dict=joint_states[topic.topic][joint.name],
                    low_dim_obs_list=low_dim_obs_names,
                    actions_list=actions_names
                )

    # Determine where to place twist data according to configuration
    twists = {}
    if hasattr(cfg, 'twists'):
        for topic in cfg.twists:
            topics.append(topic.topic)

            twists[topic.topic] = {}

            place_sub_attributes(
                obj=topic,
                attr='linear',
                sub_attrs=['x','y','z'],
                namespace=topic.topic,
                locator_dict=twists[topic.topic],
                low_dim_obs_list=low_dim_obs_names,
                actions_list=actions_names
            )
            place_sub_attributes(
                obj=topic,
                attr='angular',
                sub_attrs=['x','y','z'],
                namespace=topic.topic,
                locator_dict=twists[topic.topic],
                low_dim_obs_list=low_dim_obs_names,
                actions_list=actions_names
            )

    # Collect data from bags (each is an episode)
    for bag in pathlib.Path(cfg.input_path).iterdir():
        # Skip anything that isn't a directory since this isn't a ROS bag
        if not bag.is_dir():
            continue

        print(f'Converting episode {replay_buffer.n_episodes + 1}:\t{bag.name}', end='')

        # Decimate data according to configured rate
        episode_data = decimate(str(bag), topics, cfg.rate)

        print(f'\t{len(episode_data)} data frames\t\t{len(episode_data)/cfg.rate} sec')

        # Map from data types to output data arrays
        data_arrays = {
            DataType.TIMESTEP: np.zeros((len(episode_data), 1), dtype=np.float32),
            DataType.LOW_DIM_OBS: np.zeros((len(episode_data), len(low_dim_obs_names)), dtype=np.float32),
            DataType.ACTION: np.zeros((len(episode_data), len(actions_names)), dtype=np.float32),
        }

        # interpret all data for each timestep
        for t in range(len(episode_data)):
            data_frame = episode_data[t]

            topics_ndx_offset = 0

            # PROCESS JOINT STATES
            for i in range(len(joint_states.keys())):
                # Index in topics and data frame
                ndx = topics_ndx_offset + i

                # Map from joint names to where the data should be placed
                joint_state_map = joint_states[topics[ndx]]

                # Get message from data frame
                joint_state_msg: sensor_msgs.msg.JointState = data_frame[ndx]
                
                for j, joint in enumerate(joint_state_msg.name):
                    if not joint in joint_state_map.keys():
                        continue

                    # For each attribute, a locator should be in the joint_state_map[joint]
                    # dictionary.
                    # locator.dtype gives the data type (action, low_dim_obs, etc) which
                    # determines which data_array the data will go into
                    # locator.ndx gives the index in that data_array this value should go
                    if 'position' in joint_state_map[joint].keys():
                        locator = joint_state_map[joint]['position']
                        data_arrays[locator.dtype][t, locator.ndx] = joint_state_msg.position[j]
                    if 'velocity' in joint_state_map[joint].keys():
                        locator = joint_state_map[joint]['velocity']
                        data_arrays[locator.dtype][t, locator.ndx] = joint_state_msg.velocity[j]
                    if 'effort' in joint_state_map[joint].keys():
                        locator = joint_state_map[joint]['effort']
                        data_arrays[locator.dtype][t, locator.ndx] = joint_state_msg.effort[j]
            
            # Offset the number of topics by the number of joint_states topics
            topics_ndx_offset += len(joint_states.keys())

            # PROCESS TWISTS
            for i in range(len(twists.keys())):
                ndx = topics_ndx_offset + i

                # Map from twists to where the data should be placed
                twist_map = twists[topics[ndx]]

                # Get message from data frame
                twist_msg: geometry_msgs.msg.Twist = data_frame[ndx]

                for vector in ['linear', 'angular']:
                    if not vector in twist_map.keys():
                        continue
                    vector_msg = getattr(twist_msg, vector)

                    for axis in ['x', 'y', 'z']:
                        if not axis in twist_map[vector].keys():
                            continue
                        locator = twist_map[vector][axis]
                        data_arrays[locator.dtype][t, locator.ndx] = getattr(vector_msg, axis)

            # Offset the number of topics by the number of twists topics
            topics_ndx_offset += len(twists.keys())


            # Process timesteps
            data_arrays[DataType.TIMESTEP][t] = data_frame[-1]

        # Format episode data for inclusion in replay buffer
        episode_data_dict = {
            'timestep': data_arrays[DataType.TIMESTEP],
            'low_dim_obs': data_arrays[DataType.LOW_DIM_OBS],
            'action': data_arrays[DataType.ACTION],
        }

        # Add episode to replay buffer
        replay_buffer.add_episode(episode_data_dict, compressors='disk')

    # TODO output order somehow

    print(f'Converted {replay_buffer.n_episodes} episodes.')

if __name__ == "__main__":
    main()