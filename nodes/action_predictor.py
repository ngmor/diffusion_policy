#!/usr/bin/env python3

import torch
import dill
import hydra

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter


class ActionPredictor(Node):
    def __init__(self):
        super().__init__('action_predictor')
        # PARAMETERS
        self.declare_parameter('checkpoint_path', 'src/diffusion_policy/data/outputs/2023.10.31/16.54.28_train_diffusion_unet_image_omnid_image/checkpoints/latest.ckpt')
        # self.declare_parameter('checkpoint_path', 'src/diffusion_policy/data/outputs/2023.10.31/17.31.49_train_diffusion_unet_lowdim_omnid_lowdim/checkpoints/latest.ckpt')
        checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        
        self.declare_parameter('num_inference_steps', 16)
        self.num_inference_steps = self.get_parameter('num_inference_steps').get_parameter_value().integer_value

        # Load payload/workspace
        self.payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        self.cfg = self.payload['cfg']
        workspace_cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace: BaseWorkspace = workspace_cls(self.cfg)
        self.workspace.load_payload(self.payload, exclude_keys=None, include_keys=None)

        # Load model
        # Type hint is fine even if the policy is only low dim
        self.policy: BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU

        self.policy.eval().to(self.device)

        # Set interface parameters
        if 'DDIM' in self.cfg.policy.noise_scheduler._target_:
            self.policy.num_inference_steps = self.num_inference_steps
        else:
            self.get_logger().info('DDIM noise scheduler not used, ignoring inference steps parameter')
            self.num_inference_steps = self.policy.num_inference_steps
            self.set_parameters([Parameter(name='num_inference_steps', value=self.num_inference_steps)])

        self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1

        self.get_logger().info('Successfully loaded model.')


def main(args=None):
    rclpy.init(args=args)
    node = ActionPredictor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()