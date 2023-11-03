from typing import Any
from enum import Enum
from enum import auto as enum_auto
import pathlib
import numpy as np
import json
import click
import hydra
from omegaconf import OmegaConf

import sensor_msgs.msg
import geometry_msgs.msg
from cv_bridge import CvBridge
import cv2

from diffusion_policy.common.replay_buffer import ReplayBuffer

from omnid_bag import decimate

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

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
    # Check if the object has this attribute
    if hasattr(obj, attr):
        full_name = namespace + '.' + attr

        # In the locator dict, create a list of locations where this attribute's data is to be stored
        locator_dict[attr] = []

        # Locations are based on the specified data type from the config
        for dtype_str in getattr(obj, attr):
            if dtype_str == 'low_dim':
                dtype = DataType.LOW_DIM_OBS
                out_list = low_dim_obs_list
            elif dtype_str == 'action':
                dtype = DataType.ACTION
                out_list = actions_list
            else:
                raise Exception(f'Unexpected value for data type: {full_name}. Should be \'low_dim\' or \'action\'')
        
            # In the locator dict, store information about the data type (which tells what list this
            # should go into) as well as the index of this information in the list
            locator_dict[attr].append(DataLocator(dtype, len(out_list)))
            # Keep a record of what each in the output list is
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

@click.command()
@click.option('-c', '--config', required=True)
def main(config):
    hydra.initialize(config_path='../config', job_name='convert_bag', version_base=None)
    cfg = hydra.compose(
        config_name='train_diffusion_unet_real_image_workspace', # Doesn't matter which this is
        overrides=[
            'task=omnid_image',
            f'task/data_conversion={config}',
        ]
    ).task.data_conversion

    # Init replay buffer for handling output
    output_dir = pathlib.Path(cfg.output_path)
    assert output_dir.parent.is_dir()
    zarr_path = str(output_dir.joinpath(output_dir.name + '.zarr').absolute())
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode='w')

    # Determine what data will go where in the inputs to the model
    topics = []
    low_dim_obs_names = []
    actions_names = []
    camera_names = []

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

    # Find image topics in configuration
    camera_shapes = {}
    if hasattr(cfg, 'images'):
        for topic in cfg.images:
            topics.append(topic.topic)
            split_name = topic.topic.split('/')
            camera_name = split_name[1] if split_name[0] == '' else split_name[0]
            camera_names.append(camera_name)
            # Camera shapes are specified as CHW (for use in training)
            # but the data is stored as HWC, hence the reformat here
            camera_shapes[camera_name] = [topic.shape[1], topic.shape[2], topic.shape[0]]

    bridge = CvBridge()

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

        # Init camera data arrays
        # TODO is there a better way to handle this to save RAM?
        camera_data_arrays = {}
        for camera, shape in camera_shapes.items():
            camera_data_arrays[camera] = np.zeros((len(episode_data), *shape), dtype=np.uint8)

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

                    # For each attribute, a locator array should be in the joint_state_map[joint]
                    # dictionary.
                    # locator.dtype gives the data type (action, low_dim_obs, etc) which
                    # determines which data_array the data will go into
                    # locator.ndx gives the index in that data_array this value should go
                    if 'position' in joint_state_map[joint].keys():
                        for locator in joint_state_map[joint]['position']:
                            data_arrays[locator.dtype][t, locator.ndx] = joint_state_msg.position[j]
                    if 'velocity' in joint_state_map[joint].keys():
                        for locator in joint_state_map[joint]['velocity']:
                            data_arrays[locator.dtype][t, locator.ndx] = joint_state_msg.velocity[j]
                    if 'effort' in joint_state_map[joint].keys():
                        for locator in joint_state_map[joint]['effort']:
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
                        for locator in twist_map[vector][axis]:
                            data_arrays[locator.dtype][t, locator.ndx] = getattr(vector_msg, axis)

            # Offset the number of topics by the number of twists topics
            topics_ndx_offset += len(twists.keys())

            # PROCESS IMAGES
            for i, camera_name in enumerate(camera_names):
                ndx = topics_ndx_offset + i

                # Get camera shape
                shape = camera_shapes[camera_name]

                # Get message from data frame
                image_msg: sensor_msgs.msg.CompressedImage = data_frame[ndx]

                # Convert to cv2 image
                image = bridge.compressed_imgmsg_to_cv2(image_msg)
                
                # Resize
                image = cv2.resize(image, (shape[1], shape[0]))

                # Store
                camera_data_arrays[camera_name][t] = image


            # Process timesteps
            data_arrays[DataType.TIMESTEP][t] = data_frame[-1]

        # Format episode data for inclusion in replay buffer
        episode_data_dict = {
            'timestep': data_arrays[DataType.TIMESTEP],
            'low_dim_obs': data_arrays[DataType.LOW_DIM_OBS],
            'action': data_arrays[DataType.ACTION],
        }

        # TODO remove
        # for t in range(len(episode_data)):
        #     for camera, data in camera_data_arrays.items():
        #         cv2.imshow(camera, data[t])
        #     cv2.waitKey(1)
        #     import time
        #     time.sleep(1.0/cfg.rate)

        # Add camera data
        episode_data_dict.update(camera_data_arrays)

        # Add episode to replay buffer
        replay_buffer.add_episode(episode_data_dict, compressors='disk')

    # TODO could be good to output this in a better format
    out_format = {
        'actions': actions_names,
        'low_dim_obs': low_dim_obs_names,
        'cameras': {}
    }
    for camera, shape in camera_shapes.items():
        out_format['cameras'][camera] = list(shape)

    with open(str(output_dir.joinpath(output_dir.name + '.format.json').absolute()), 'w') \
    as out_format_file:
        json.dump(out_format, out_format_file, indent=4)

    print(f'Converted {replay_buffer.n_episodes} episodes.')

if __name__ == "__main__":
    main()