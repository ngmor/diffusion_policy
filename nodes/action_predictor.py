#!/usr/bin/env python3

import torch
import dill
import hydra

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.ros_util import ROSDataConverter

import rclpy
from rclpy.node import Node
import message_filters

class ActionPredictor(Node):
    def __init__(self):
        super().__init__('action_predictor')
        # PARAMETERS
        self.declare_parameter('checkpoint_path', 'src/diffusion_policy/data/outputs/2023.11.03/01.22.15_train_diffusion_unet_image_omnid_image/checkpoints/epoch=0000-train_loss=1.343.ckpt')
        # self.declare_parameter('checkpoint_path', 'src/diffusion_policy/data/outputs/2023.11.03/01.22.46_train_diffusion_unet_lowdim_omnid_lowdim/checkpoints/latest.ckpt')
        checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value

        # TODO method of disabling specific cameras

        # # Load payload/workspace
        self.payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        self.cfg = self.payload['cfg']
        self.low_dim = 'lowdim' in self.cfg.task
        self.observation_rate = self.cfg.task.data_conversion.rate
        self.observation_period = 1.0 / self.observation_rate

        # TODO uncomment
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

        self.data_converter = ROSDataConverter(
            cfg=self.cfg.task.data_conversion,
            disable_cameras=self.low_dim
        )

        topics, topic_info = self.data_converter.get_topics_and_info()

        input_data_subscribers = []

        for topic in topics:
            input_data_subscribers.append(message_filters.Subscriber(
                self,
                topic_info[topic]['type'],
                topic
            ))
        self.ats = message_filters.ApproximateTimeSynchronizer(
            input_data_subscribers,
            self.cfg.n_obs_steps,
            self.observation_period,
            allow_headerless=True
        )
        self.ats.registerCallback(self.synchronized_observation_callback)

        self.last_obs_callback_time = self.get_clock().now()
        self.input_data_queue = []
        self.get_logger().info(f'Synchronizing at {self.observation_rate} Hz')


    def synchronized_observation_callback(self, *msgs):
        receive_time = self.get_clock().now()
        if (receive_time - self.last_obs_callback_time).nanoseconds / 1.0e9 \
        < self.observation_period:
            return

        if len(self.input_data_queue) >= self.cfg.n_obs_steps:
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