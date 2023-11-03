import pathlib
import json
import click
import hydra
from omegaconf import OmegaConf

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.ros_util import ROSDataConverter

from omnid_bag import decimate

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

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

    # Initialize data converter with config
    data_converter = ROSDataConverter(cfg)

    # Collect data from bags (each is an episode)
    for bag in pathlib.Path(cfg.input_path).iterdir():
        # Skip anything that isn't a directory since this isn't a ROS bag
        if not bag.is_dir():
            continue

        print(f'Converting episode {replay_buffer.n_episodes + 1}:\t{bag.name}', end='')

        # Decimate data according to configured rate
        episode_data = decimate(str(bag), data_converter.get_topics(), cfg.rate)

        print(f'\t{len(episode_data)} data frames\t\t{len(episode_data)/cfg.rate} sec')

        # Convert data
        converted_data = data_converter.convert_data_frames(episode_data)

        # TODO remove
        # import cv2
        # import time
        # while(True):
        #     for t in range(len(episode_data)):
        #         for camera in data_converter.get_camera_names():
        #             cv2.imshow(camera, converted_data[camera][t])
        #         cv2.waitKey(1)
        #         time.sleep(1.0/cfg.rate)

        for t in range(len(episode_data)):
            print(converted_data['action'][t])
            print(converted_data['low_dim_obs'][t])

        # Add episode to replay buffer
        replay_buffer.add_episode(converted_data, compressors='disk')

    # TODO could be good to output this in a better format
    with open(str(output_dir.joinpath(output_dir.name + '.format.json').absolute()), 'w') \
    as out_format_file:
        json.dump(data_converter.get_format(), out_format_file, indent=4)

    print(f'Converted {replay_buffer.n_episodes} episodes.')

if __name__ == "__main__":
    main()