import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.common.replay_buffer import ReplayBuffer

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

PACKAGE_PATH = pathlib.Path(__file__).parent.parent.parent

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

if __name__ == "__main__":
    main()