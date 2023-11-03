#!/usr/bin/env python3

import torch
import dill
import hydra

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import sensor_msgs.msg
import message_filters

class ActionPredictor(Node):
    def __init__(self):
        super().__init__('action_predictor')
        # PARAMETERS
        self.declare_parameter('checkpoint_path', 'src/diffusion_policy/data/outputs/2023.10.31/16.54.28_train_diffusion_unet_image_omnid_image/checkpoints/latest.ckpt')
        # self.declare_parameter('checkpoint_path', 'src/diffusion_policy/data/outputs/2023.10.31/17.31.49_train_diffusion_unet_lowdim_omnid_lowdim/checkpoints/latest.ckpt')
        checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value

        self.declare_parameter('num_inference_steps', 16)
        self.num_inference_steps = self.get_parameter('num_inference_steps').get_parameter_value().integer_value

        self.declare_parameter('observation_rate_rate', 20.0)
        self.observation_rate_rate = self.get_parameter('observation_rate_rate').get_parameter_value().double_value
        self.observation_rate_period = 1.0 / self.observation_rate_rate

        # TODO Should probably be some factor slower the observation_rate_rate
        # based on the receding horizon control
        # (8 times slower here)
        self.declare_parameter('inference_rate', 2.5)
        self.inference_rate = self.get_parameter('inference_rate').get_parameter_value().double_value
        self.inference_period = 1.0 / self.inference_rate

        self.declare_parameter('overhead_camera.enable', True)
        self.overhead_camera_enable = self.get_parameter('overhead_camera.enable').get_parameter_value().bool_value

        self.declare_parameter('horizontal_camera.enable', True)
        self.horizontal_camera_enable = self.get_parameter('horizontal_camera.enable').get_parameter_value().bool_value

        self.declare_parameter('onboard_camera.enable', True)
        self.onboard_camera_enable = self.get_parameter('onboard_camera.enable').get_parameter_value().bool_value

        # FILTERS FOR PREDICTION DATA
        input_data_subscribers = []

        if self.overhead_camera_enable:
            input_data_subscribers.append(message_filters.Subscriber(
                self,
                sensor_msgs.msg.CompressedImage,
                "/overhead_camera/color/image_raw/compressed"
            ))
        if self.horizontal_camera_enable:
            input_data_subscribers.append(message_filters.Subscriber(
                self,
                sensor_msgs.msg.CompressedImage,
                "/horizontal_camera/color/image_raw/compressed"
            ))
        if self.onboard_camera_enable:
            input_data_subscribers.append(message_filters.Subscriber(
                self,
                sensor_msgs.msg.CompressedImage,
                "/onboard_camera/color/image_raw/compressed"
            ))
        self.ats = message_filters.ApproximateTimeSynchronizer(
            input_data_subscribers,
            10, # queue size - TODO make correspond to model num observations
            self.observation_rate_period
        )
        self.ats.registerCallback(self.synchronized_observation_callback)

        # TODO rename and formalize
        self.last_obs_callback_time = self.get_clock().now()

        self.input_data_queue = []

        # TODO uncomment
        # # Load payload/workspace
        # self.payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        # self.cfg = self.payload['cfg']
        # workspace_cls = hydra.utils.get_class(self.cfg._target_)
        # self.workspace: BaseWorkspace = workspace_cls(self.cfg)
        # self.workspace.load_payload(self.payload, exclude_keys=None, include_keys=None)

        # # Load model
        # # Type hint is fine even if the policy is only low dim
        # self.policy: BaseImagePolicy = self.workspace.model
        # if self.cfg.training.use_ema:
        #     self.policy = self.workspace.ema_model

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU

        # self.policy.eval().to(self.device)

        # # Set interface parameters
        # if 'DDIM' in self.cfg.policy.noise_scheduler._target_:
        #     self.policy.num_inference_steps = self.num_inference_steps
        # else:
        #     self.get_logger().info('DDIM noise scheduler not used, ignoring inference steps parameter')
        #     self.num_inference_steps = self.policy.num_inference_steps
        #     self.set_parameters([Parameter(name='num_inference_steps', value=self.num_inference_steps)])

        # self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1

        # self.get_logger().info('Successfully loaded model.')

    def synchronized_observation_callback(self, *msgs):
        receive_time = self.get_clock().now()
        if (receive_time - self.last_obs_callback_time).nanoseconds / 1.0e9 \
        < self.observation_rate_period:
            return

        # queue size - TODO make correspond to model num observations
        if len(self.input_data_queue) >= 10:
            self.input_data_queue = self.input_data_queue[1:] + [msgs]
        else:
            self.input_data_queue.append(msgs)

        print(1.0e9 / (receive_time - self.last_obs_callback_time).nanoseconds, len(self.input_data_queue))
        self.last_obs_callback_time = receive_time

def main(args=None):
    rclpy.init(args=args)
    node = ActionPredictor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()