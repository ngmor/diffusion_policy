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
    """Enumeration for what type of data (in relation to diffusion policy model inputs/outputs) a certain value is."""
    TIMESTEP = enum_auto()
    ACTION = enum_auto()
    LOW_DIM_OBS = enum_auto()
    IMG = enum_auto()

class DataLocator:
    """Class which holds information about where to find/place data in input/output arrays for diffusion policy models."""
    def __init__(self, dtype: DataType, ndx: int):
        """
        Initialize data locator object.

        Args:
            dtype (DataType): the type of data for a certain value
            ndx (int): the index in that data type's output array where that certain value can be found
        """
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

class CameraShape:
    """Helper class for storing the dimensions of camera images and returning them in any requested format."""
    def __init__(self, C, H, W):
        """
        Initialize camera shape object.

        Args:
            C (int): number of channels in the image
            H (int): height of the image in pixels
            W (int): width of the image in pixels
        """
        self.C = C
        self.H = H
        self.W = W
    def as_tuple(self, format):
        """
        Return image shape as a tuple in the requested format.

        Args:
            format (str): a string consisting of the characters 'C', 'H', and 'W' in any
                arbitrary order. This order determines the order of the output tuple

        Returns:
            Tuple[int]: image shape in the requested format
        """
        out = list()
        for dim in format:
            out.append(getattr(self, dim))
        return tuple(out)

class ROSDataConverter:
    """Converts input data frames of ROS messages into np arrays of the format expected by a diffusion policy model."""

    def __init__(
        self,
        cfg,
        exclude_cameras: List[str]=[],
        camera_format='HWC',
        low_dim_obs_key='low_dim_obs'
    ):
        """
        Initialize/configure data converter.

        Args:
            cfg: "data_conversion" config from hydra. This is the "task.data_conversion" member of
                the full hydra config that the diffusion_policy repo uses
            exclude_cameras (List[str], optional): Cameras to exclude from the data. If the list
                contains '*', all cameras will be excluded. Defaults to empty.
            camera_format (str, optional): a string containing the characters 'H' (height),
                'W' (width), and 'C' (channels). This defines the output shape of the image data.
                Defaults to 'HWC'.
            low_dim_obs_key (str, optional): the key of the low dim observation data in the
                conversion output. Defaults to 'low_dim_obs'.
        """
        self.cfg = cfg
        # TODO - how to handle not having input actions(?)

        # Initialize/store class configuration variables
        self.topics = []
        self.topic_info = {}
        self.actions_names = []
        self.low_dim_obs_names = []
        self.camera_names = []
        self.camera_format = camera_format
        self.low_dim_obs_key = low_dim_obs_key

        # Determine what topics will go where in output data based on configuration
        # Determine where to place joint state data according to configuration
        self.joint_states = {}
        if hasattr(cfg, 'joint_states'):
            for topic in cfg.joint_states:
                self._add_topic(topic.topic, sensor_msgs.msg.JointState)

                self.joint_states[topic.topic] = {}

                for joint in topic.joints:
                    self.joint_states[topic.topic][joint.name] = {}

                    namespace = topic.topic + '.' + joint.name

                    self._place_scalar_attribute(
                        joint,
                        'position',
                        namespace,
                        self.joint_states[topic.topic][joint.name]
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
                self._add_topic(topic.topic, geometry_msgs.msg.Twist)

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
        if hasattr(cfg, 'images') and not '*' in exclude_cameras:
            for topic in cfg.images:
                split_name = topic.topic.split('/')
                camera_name = split_name[1] if split_name[0] == '' else split_name[0]
                if camera_name in exclude_cameras:
                    # Exclude specific cameras
                    continue
                self._add_topic(topic.topic, sensor_msgs.msg.CompressedImage)
                self.camera_names.append(camera_name)
                # Topics are stored in the config as CHW
                self.camera_shapes[camera_name] = CameraShape(*topic.shape)

        self.bridge = CvBridge()

    def _add_topic(self, topic, msg_type):
        """Add a topic to the list of input topics and determine its index in that list."""
        self.topic_info[topic] = {
            'ndx': len(self.topics),
            'type': msg_type
        }
        self.topics.append(topic)

    def _place_scalar_attribute(
        self,
        obj: Any,
        attr: str,
        namespace: str,
        locator_dict: dict
    ):
        """
        Determines the location in the output data for a scalar value in ROS message.

        The "data_conversion" config defines how ROS messages are parsed into low dim observations
        and actions. A scalar value in a ROS message (ex: a joint's effort) may be placed into one
        or both of the low dim observations array (an input to the diffusion policy model) and the
        actions array (the output of the diffusion policy model). This function reads the scalar
        attribute (attr) of an object (obj) in the config structure, parses it, and determines
        the placement of this scalar value in those arrays accordingly.

        Args:
            obj (Any): the member of the config structure to search for the attribute in.
            attr (str): the name of the attribute, which may or may not exist in the config
                structure. If it is there, it will be included in the arrays where indicated by its
                value
            namespace (str): previous human-readable path to this config member, including the
                topic, etc. Used to compile a human-readable list of members of each array
            locator_dict (dict): the dictionary in which to store a list of DataLocator objects,
                which contains information about where the attribute is to be placed in the
                appropriate arrays

        Raises:
            Exception: if the value of the attribute is not a list consisting only of 'low_dim',
                'action', or both.
        """
        
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
        """
        Place multiple scalar sub-attributes of an attribute of an object in the config structure.

        See _place_scalar_attribute for more info on what placing an attribute does.


        Args:
            obj (Any): the member of the config structure to search for the attribute in.
            attr (str): the name of the attribute, which may or may not exist in the config
                structure.
            sub_attrs (List[str]): sub attributes of the attribute to place
            namespace (str): previous human-readable path to this config member, including the
                topic, etc. Used to compile a human-readable list of members of each array
            locator_dict (dict): the dictionary in which to store a list of DataLocator objects,
                which contains information about where the attribute is to be placed in the
                appropriate arrays
        """
        if hasattr(obj, attr):
            full_name = namespace + '.' + attr

            temp_dict = {}

            for sub_attr in sub_attrs:
                self._place_scalar_attribute(getattr(obj, attr), sub_attr, full_name, temp_dict)
            
            if len(temp_dict.keys()) > 0:
                locator_dist[attr] = temp_dict

    def convert_data_frames(self, data_frames: List):
        """
        # Convert an input list of data frames of ROS messages to the np array format of the
        inputs/outputs of the diffusion policy model.

        Args:
            data_frames (List): list of data frames to convert. Each data frame itself should be a
            list, which contains ROS messages in the proper order as determined by the configuration
            of this converter object. A list of the topics for these messages in the proper order
            can be obtained by using the get_topics method.

        Returns:
            dict: dictionary containing keys for each data type ('low_dim_obs', 'action', etc). The
                values are the converted data as np arrays.
                Note: low_dim_obs key is determined by the constructor of this class
        """

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
                (len(data_frames), *shape.as_tuple(self.camera_format)),
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
                image = cv2.resize(image, (shape.W, shape.H))

                if self.camera_format != 'HWC':
                    # Move axes to fit the requested format
                    image = np.moveaxis(
                        image,
                        [0, 1, 2],
                        [
                            self.camera_format.find('H'),
                            self.camera_format.find('W'),
                            self.camera_format.find('C')
                        ]
                    )

                # Store
                camera_data_arrays[camera_name][t] = image

            # PROCESS TIMESTEPS
            data_arrays[DataType.TIMESTEP][t] = data_frames[t][-1]

        # Format output dict
        out_dict = {
            'timestep': data_arrays[DataType.TIMESTEP],
            self.low_dim_obs_key: data_arrays[DataType.LOW_DIM_OBS],
            'action': data_arrays[DataType.ACTION],
        }

        # Add camera data
        out_dict.update(camera_data_arrays)

        return out_dict

    def get_topics(self):
        """Get ordered list of all topics from which an input data frame should have messages."""
        return self.topics
    
    def get_topic_info(self):
        """Get dictionary that maps topic names to info about the topics (index in topic list, message type)."""
        return self.topic_info

    def get_topics_and_info(self):
        """Return outputs of get_topics and get_topic_info as a tuple."""
        return (self.get_topics(), self.get_topic_info())

    def get_camera_names(self):
        """Get names of cameras."""
        return self.camera_names
    
    def get_format(self):
        """Get human-readable format of inputs/outputs of diffusion policy model."""
        out_format = {
            'action': self.actions_names,
            'low_dim_obs': self.low_dim_obs_names,
            'cameras': {}
        }
        for camera, shape in self.camera_shapes.items():
            out_format['cameras'][camera] = list(shape.as_tuple(self.camera_format))

        return out_format