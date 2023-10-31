from typing import Any

from enum import Enum
from enum import auto as enum_auto
import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.common.replay_buffer import ReplayBuffer

from omnid_bag import decimate

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

PACKAGE_PATH = pathlib.Path(__file__).parent.parent.parent

class DataType(Enum):
    ACTION = enum_auto()
    LOW_DIM_OBS = enum_auto()
    IMG = enum_auto()

class DataLocator:
    def __init__(self, dtype: DataType, ndx: int):
        self.dtype = dtype
        self.ndx = ndx
    def __repr__(self) -> str:
        return f'({self.dtype.name}, {self.ndx})'

def add_scalar_attribute(obj: Any, attr: str, namespace: str, locator_dict: dict, low_dim_obs_list: list, actions_list: list):
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
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode='a')

    # Determine what data will go where in the inputs to the model
    topics = []
    low_dim_obs = []
    actions = []

    joint_states = {}

    if hasattr(cfg, 'joint_states'):
        for topic in cfg.joint_states:
            topics.append(topic.topic)

            joint_states[topic.topic] = {}

            for joint in topic.joints:
                joint_states[topic.topic][joint.name] = {}

                namespace = topic.topic + '.' + joint.name

                add_scalar_attribute(
                    obj=joint,
                    attr='position',
                    namespace=namespace,
                    locator_dict=joint_states[topic.topic][joint.name],
                    low_dim_obs_list=low_dim_obs,
                    actions_list=actions
                )
                add_scalar_attribute(
                    obj=joint,
                    attr='velocity',
                    namespace=namespace,
                    locator_dict=joint_states[topic.topic][joint.name],
                    low_dim_obs_list=low_dim_obs,
                    actions_list=actions
                )
                add_scalar_attribute(
                    obj=joint,
                    attr='effort',
                    namespace=namespace,
                    locator_dict=joint_states[topic.topic][joint.name],
                    low_dim_obs_list=low_dim_obs,
                    actions_list=actions
                )

    # Collect data from bags (each is an episode)
    for bag in pathlib.Path(cfg.input_path).iterdir():
        # Skip anything that isn't a directory since this isn't a ROS bag
        if not bag.is_dir():
            continue
        
        # Decimate data according to configured rate
        episode_data = decimate(str(bag), topics, cfg.rate)


        # TODO remove
        break

    # print(topics)
    # print(joint_states)
    # print(low_dim_obs)
    # print(actions)

if __name__ == "__main__":
    main()