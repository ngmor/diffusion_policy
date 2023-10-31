from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

class OmnidLowdimRunner(BaseLowdimRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseLowdimPolicy):
        log = dict()
        log['test_mean_score'] = 0.0
        return log