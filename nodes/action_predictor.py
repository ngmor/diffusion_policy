#!/usr/bin/env python3

import torch
import dill
import hydra
import numpy as np

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.ros_util import ROSDataConverter

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import message_filters

class LastMessageSubscriber:
    def __init__(
        self,
        action_predictor_node: Node, # actually ActionPredictor
        topic,
        topic_type,
        topic_ndx,
    ):
        self.action_predictor_node = action_predictor_node
        self.action_predictor_node.create_subscription(
            msg_type=topic_type,
            topic=topic,
            callback=self.callback,
            qos_profile=10
        )
        self.ndx = topic_ndx
    def callback(self, msg):
        self.action_predictor_node.received[self.ndx] = True
        self.action_predictor_node.last_message[self.ndx] = msg


class ActionPredictor(Node):
    def __init__(self):
        super().__init__('action_predictor')
        # PARAMETERS
        self.declare_parameter('checkpoint_path', 'src/diffusion_policy/data/outputs/2023.11.03/01.22.15_train_diffusion_unet_image_omnid_image/checkpoints/epoch=0000-train_loss=1.343.ckpt')
        # self.declare_parameter('checkpoint_path', 'src/diffusion_policy/data/outputs/2023.11.03/01.22.46_train_diffusion_unet_lowdim_omnid_lowdim/checkpoints/epoch=0000-test_mean_score=0.000.ckpt')
        checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value

        self.declare_parameter('num_inference_diffusion_timesteps', 16)
        self.num_inference_diffusion_timesteps = self.get_parameter('num_inference_diffusion_timesteps').get_parameter_value().integer_value

        # TODO method of disabling specific cameras

        # # Load payload/workspace
        self.payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        self.cfg = self.payload['cfg']
        self.low_dim = 'lowdim' in self.cfg.task.name
        self.observation_rate = self.cfg.task.data_conversion.rate
        self.observation_period = 1.0 / self.observation_rate

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
            self.policy.num_inference_steps = self.num_inference_diffusion_timesteps
        else:
            self.get_logger().info('DDIM noise scheduler not used, ignoring inference steps parameter')
            self.num_inference_diffusion_timesteps = self.policy.num_inference_steps
            self.set_parameters([Parameter(name='num_inference_diffusion_timesteps', value=self.num_inference_diffusion_timesteps)])

        self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1

        self.get_logger().info('Successfully loaded model.')

        # Init data converter
        self.data_converter = ROSDataConverter(
            cfg=self.cfg.task.data_conversion,
            disable_cameras=self.low_dim
        )

        topics, topic_info = self.data_converter.get_topics_and_info()

        self.input_data_subscribers = []
        self.last_message = []

        for topic in topics:
            self.last_message.append(None)
            self.input_data_subscribers.append(
                LastMessageSubscriber(self, topic, topic_info[topic]['type'], topic_info[topic]['ndx'])
            )

        self.received = np.full((len(self.last_message),), False)
        self.timer_input_data = self.create_timer(self.observation_period, self.timer_observation_callback)
        self.at_least_one_received = False
        self.n_obs_received = False


        self.last_obs_callback_time = self.get_clock().now()
        self.input_data_queue = []
        self.get_logger().info(f'Synchronizing at {self.observation_rate} Hz')

    def timer_observation_callback(self):
        if np.any(~self.received):
            if not self.at_least_one_received:
                return
            for i in range(len(self.received)):
                if not self.received[i]:
                    self.get_logger().warn(f'{self.data_converter.get_topics()[i]} not received in the last observation period, using previous value')
        self.at_least_one_received = True
        self.received = np.full((len(self.last_message),), False)

        if len(self.input_data_queue) >= self.cfg.n_obs_steps:
            self.input_data_queue = self.input_data_queue[1:] + [self.last_message]
        else:
            if not self.n_obs_received:
                self.get_logger().info(f'{self.cfg.n_obs_steps} observations received, ready to begin inference.')
                self.n_obs_received = True
            self.input_data_queue.append(self.last_message)

def main(args=None):
    rclpy.init(args=args)
    node = ActionPredictor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()