from typing import Any, List
from enum import Enum
from enum import auto as enum_auto
import numpy as np
import cv2

from omegaconf import OmegaConf

from cv_bridge import CvBridge
import sensor_msgs.msg
import geometry_msgs.msg

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

class ROSDataConverter:
    def __init__(self, cfg):
        self.cfg = cfg
        # TODO - how to handle not having input actions(?)

        # Determine what topics will go where in output data
        # based on config
        self.topics = []
        self.actions_names = []
        self.low_dim_obs_names = []
        self.camera_names = []

        # Determine where to place joint state data according to configuration
        self.joint_states = {}
        if hasattr(cfg, 'joint_states'):
            for topic in cfg.joint_states:
                self.topics.append(topic.topic)

                self.joint_states[topic.topic] = {}

                for joint in topic.joints:
                    self.joint_states[topic.topic][joint.name] = {}

                    namespace = topic.topic + '.' + joint.name

                    self._place_scalar_attribute(
                        joint,
                        'position',
                        namespace, self.joint_states[topic.topic][joint.name]
                    )
                    self._place_scalar_attribute(
                        joint,
                        'velocity',
                        namespace,
                        self.joint_states[topic.topic][joint.name]
                    )
                    self._place_scalar_attribute(
                        joint,
                        'effort',
                        namespace,
                        self.joint_states[topic.topic][joint.name]
                    )

        # Determine where to place twist data according to configuration
        self.twists = {}
        if hasattr(cfg, 'twists'):
            for topic in cfg.twists:
                self.topics.append(topic.topic)

                self.twists[topic.topic] = {}

                self._place_sub_attributes(
                    topic,
                    'linear',
                    ['x', 'y', 'z'],
                    topic.topic,
                    self.twists[topic.topic]
                )
                self._place_sub_attributes(
                    topic,
                    'angular',
                    ['x', 'y', 'z'],
                    topic.topic,
                    self.twists[topic.topic]
                )

        # Find image topics in configuration
        self.camera_shapes = {}
        if hasattr(cfg, 'images'):
            for topic in cfg.images:
                self.topics.append(topic.topic)
                split_name = topic.topic.split('/')
                camera_name = split_name[1] if split_name[0] == '' else split_name[0]
                self.camera_names.append(camera_name)
                # Camera shapes are specified as CHW (for use in training)
                # but the data is stored as HWC, hence the reformat here
                self.camera_shapes[camera_name] = [topic.shape[1], topic.shape[2], topic.shape[0]]

        self.bridge = CvBridge()


    def _place_scalar_attribute(
        self,
        obj: Any,
        attr: str,
        namespace: str,
        locator_dict: dict
    ):
        # Check if the object has this attribute
        if hasattr(obj, attr):
            full_name = namespace + '.' + attr
            
            # In the locator dict, create a list of locations where this
            # attribute's data is to be stored
            locator_dict[attr] = []

            # Locations are based on the specified data type from the config
            for dtype_str in getattr(obj, attr):
                if dtype_str == 'low_dim':
                    dtype = DataType.LOW_DIM_OBS
                    out_list = self.low_dim_obs_names
                elif dtype_str == 'action':
                    dtype = DataType.ACTION
                    out_list = self.actions_names
                else:
                    raise Exception(
                        f'Unexpected value for data type: {full_name}.'
                        ' Should be \'low_dim\' or \'action\''
                    )

                # In the locator dict, store information about the data type
                # (which tells what list this should go into)
                # as well as the index of this information in the list
                locator_dict[attr].append(DataLocator(dtype, len(out_list)))
                out_list.append(full_name)

    # TODO make this recursive?
    def _place_sub_attributes(
        self,
        obj: Any,
        attr: str,
        sub_attrs: List[str],
        namespace: str,
        locator_dist: dict
    ):
        if hasattr(obj, attr):
            full_name = namespace + '.' + attr

            temp_dict = {}

            for sub_attr in sub_attrs:
                self._place_scalar_attribute(getattr(obj, attr), sub_attr, full_name, temp_dict)
            
            if len(temp_dict.keys()) > 0:
                locator_dist[attr] = temp_dict

    def convert_data_frames(self, data_frames: List):
        # Map from data types to output data arrays
        data_arrays = {
            DataType.TIMESTEP: np.zeros(
                (len(data_frames), 1),
                dtype=np.float32
            ),
            DataType.LOW_DIM_OBS: np.zeros(
                (len(data_frames), len(self.low_dim_obs_names)),
                dtype=np.float32
            ),
            DataType.ACTION: np.zeros(
                (len(data_frames), len(self.actions_names)),
                dtype=np.float32
            )
        }

        # Init camera data arrays
        # TODO is there a better way to handle this to save RAM?
        camera_data_arrays = {}
        for camera, shape in self.camera_shapes.items():
            camera_data_arrays[camera] = np.zeros(
                (len(data_frames), *shape),
                dtype=np.uint8
            )

        # interpret all data for each frame
        for t in range(len(data_frames)):
            topics_ndx_offset = 0

            # PROCESS JOINT STATES
            for i in range(len(self.joint_states.keys())):
                # Index in topics list and in data frame
                ndx = topics_ndx_offset + i

                # Map from joint names to where the data should be placed
                joint_state_map = self.joint_states[self.topics[ndx]]

                # Get message from data frame
                joint_state_msg: sensor_msgs.msg.JointState = data_frames[t][ndx]

                for j, joint in enumerate(joint_state_msg.name):
                    if not joint in joint_state_map.keys():
                        continue

                # For each attribute, a locator array should be in the
                # joint_state_map[joint] dictionary.
                # locator.dtype: the data type (action, low_dim_obs, etc).
                # which determines which data_array the data will go to
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
            topics_ndx_offset += len(self.joint_states.keys())

            # PROCESS TWISTS
            for i in range(len(self.twists.keys())):
                ndx = topics_ndx_offset + i

                # Map from twists to where the data should be placed
                twist_map = self.twists[self.topics[ndx]]

                # Get message from data frame
                twist_msg: geometry_msgs.msg.Twist = data_frames[t][ndx]

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
            topics_ndx_offset += len(self.twists.keys())
        
            # PROCESS IMAGES
            for i, camera_name in enumerate(self.camera_names):
                ndx = topics_ndx_offset + i

                # Get campera shape
                shape = self.camera_shapes[camera_name]

                # Get message from data frame
                image_msg: sensor_msgs.msg.CompressedImage = data_frames[t][ndx]

                # Convert to cv2 image
                image = self.bridge.compressed_imgmsg_to_cv2(image_msg)

                # Resize
                image = cv2.resize(image, (shape[1], shape[0]))

                # Store
                camera_data_arrays[camera_name][t] = image

            # PROCESS TIMESTEPS
            data_arrays[DataType.TIMESTEP][t] = data_frames[t][-1]

        # Format output dict
        out_dict = {
            'timestep': data_arrays[DataType.TIMESTEP],
            'low_dim_obs': data_arrays[DataType.LOW_DIM_OBS],
            'action': data_arrays[DataType.ACTION],
        }

        # Add camera data
        out_dict.update(camera_data_arrays)

        return out_dict

    def get_topics(self):
        return self.topics

    def get_camera_names(self):
        return self.camera_names
    
    def get_format(self):
        out_format = {
            'action': self.actions_names,
            'low_dim_obs': self.low_dim_obs_names,
            'cameras': {}
        }
        for camera, shape in self.camera_shapes.items():
            out_format['cameras'][camera] = list(shape)
        
        return out_format