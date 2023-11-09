#!/usr/bin/env python3

import torch
import dill
import hydra
import numpy as np
import threading

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.ros_util import ROSDataConverter
from diffusion_policy.common.pytorch_util import dict_apply

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import geometry_msgs.msg
import std_srvs.srv
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
        self.action_predictor_node.obs_received[self.ndx] = True
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

        self.declare_parameter('num_actions_taken', 8)
        self.num_actions_taken = self.get_parameter('num_actions_taken').get_parameter_value().integer_value

        # TODO method of disabling specific cameras

        # PUBLISHERS
        self.pub_action = self.create_publisher(geometry_msgs.msg.Wrench, '/omnid1/delta/additional_force', 10)

        # SERVICE SERVERS
        self.srv_start_inference = self.create_service(std_srvs.srv.Empty, 'start_inference', self.srv_start_inference_callback)
        self.srv_stop_inference = self.create_service(std_srvs.srv.Empty, 'stop_inference', self.srv_stop_inference_callback)
        self.srv_start_action = self.create_service(std_srvs.srv.Empty, 'start_action', self.srv_start_action_callback)
        self.srv_stop_action = self.create_service(std_srvs.srv.Empty, 'stop_action', self.srv_stop_action_callback)
        self.enable_inference = True
        self.enable_action = False


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

        if self.num_actions_taken > self.policy.n_action_steps:
            self.get_logger().warn(
                'Num actions taken is larger than horizon, clamping to'
                f' {self.policy.n_action_steps} actions'
            )
            self.num_actions_taken = self.policy.n_action_steps
            self.set_parameters([Parameter(name='num_actions_taken', value=self.num_actions_taken)])

        self.get_logger().info('Successfully loaded model.')

        # Init data converter
        if self.low_dim:
            cameras_to_exclude = ['*']
        else:
            cameras_to_exclude = [] # TODO add parameter for this
        self.data_converter = ROSDataConverter(
            cfg=self.cfg.task.data_conversion,
            exclude_cameras=cameras_to_exclude,
            camera_format='CHW' # for passing into model
        )

        topics, topic_info = self.data_converter.get_topics_and_info()

        self.input_data_subscribers = []
        self.last_message = []

        for topic in topics:
            self.last_message.append(None)
            self.input_data_subscribers.append(
                LastMessageSubscriber(self, topic, topic_info[topic]['type'], topic_info[topic]['ndx'])
            )
        self.last_message.append(None) # For timestep

        self.timer = self.create_timer(self.observation_period, self.timer_callback)

        self.reset_obs_received()
        self.at_least_one_of_each_obs_received = False
        self.n_obs_received = False
        self.obs_data_queue = []
        self.obs_data_mutex = threading.Lock()

        self.inference_counter = self.num_actions_taken
        self.inference_thread = threading.Thread()

        self.action_data_mutex = threading.Lock()
        self.action_counter = 0
        self.action_array = []

        self.get_logger().info(f'Synchronizing at {self.observation_rate} Hz')

    def reset_obs_received(self):
        self.obs_received = np.full((len(self.last_message) - 1,), False)

    def srv_start_inference_callback(self, request, response):
        self.enable_inference = True
        self.inference_counter = self.num_actions_taken  # immediately trigger inference

    def srv_stop_inference_callback(self, request, response):
        self.enable_inference = False
        self.srv_stop_action_callback(None, None)

    def srv_start_action_callback(self, request, response):
        if self.enable_inference:
            self.enable_action = True
    
    def srv_stop_action_callback(self, request, response):
        self.enable_action = False
        self.action_array = []
        self.action_counter = 0

    def timer_callback(self):
        if np.any(~self.obs_received):
            if not self.at_least_one_of_each_obs_received:
                # we haven't received one of each observation yet, no point in continuing
                return
            for i in range(len(self.obs_received)):
                if not self.obs_received[i]:
                    self.get_logger().warn(f'{self.data_converter.get_topics()[i]} not received in the last observation period, using previous value')
        self.at_least_one_of_each_obs_received = True
        self.reset_obs_received()


        # Lock mutex to modify observation data queue
        with self.obs_data_mutex:
            self.last_message[-1] = self.get_clock().now().nanoseconds
            if len(self.obs_data_queue) < self.cfg.n_obs_steps:
                self.obs_data_queue.append(self.last_message)
                # If we don't have the number of observations required for an inference,
                # do not continue
                return
            else:
                if not self.n_obs_received:
                    self.get_logger().info(
                        f'{self.cfg.n_obs_steps} observations received, ready to begin inference.'
                    )
                    self.n_obs_received = True
                # Only retain the number of observations required for an inference
                self.obs_data_queue = self.obs_data_queue[1:] + [self.last_message]

        if not self.enable_inference:
            return

        self.inference_counter += 1

        if not self.inference_thread.is_alive():
            # If it's time to infer again, start thread
            if self.inference_counter > self.num_actions_taken:
                self.inference_counter = 0
                self.inference_thread = threading.Thread(target=self.infer)
                self.inference_thread.start()


        if not self.enable_action:
            return

        # Perform actions from past inferences
        with self.action_data_mutex:
            if self.action_counter < len(self.action_array):
                print(f'Performing action {self.action_counter}: {self.action_array[self.action_counter]}')
                msg = geometry_msgs.msg.Wrench()
                msg.force.x = float(self.action_array[self.action_counter][0])
                msg.force.y = float(self.action_array[self.action_counter][1])
                self.action_counter += 1


    def infer(self):
        self.get_logger().info('Starting inference')
        # TODO remove
        import time
        start = time.time()

        # Convert observation data from ROS messages to np arrays
        with self.obs_data_mutex:
            obs_data = self.data_converter.convert_data_frames(self.obs_data_queue)
        
        if 'action' in obs_data.keys():
            del obs_data['action']
        if 'timestep' in obs_data.keys():
            del obs_data['timestep']

        # Convert to torch Tensors of the right shape
        obs_data_tensors = dict_apply(
            obs_data,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )

        # TODO remove
        # for key, value in obs_data_tensors.items():
        #     print(key, type(value), value.shape)

        # Perform inference
        with torch.no_grad():
            result = self.policy.predict_action(obs_data_tensors)

        # TODO remove
        # for key, value in result.items():
        #     print(key, type(value), value.shape)

        with self.action_data_mutex:
            self.action_array = result['action'][0].detach().to('cpu').numpy()
            self.action_counter = 0
        self.get_logger().info('Inference complete')
        self.get_logger().info(f'Time: {time.time() - start} sec')



def main(args=None):
    rclpy.init(args=args)
    node = ActionPredictor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()