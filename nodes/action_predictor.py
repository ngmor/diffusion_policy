#!/usr/bin/env python3

import torch
import dill
import hydra
import numpy as np
import threading
import yaml
from enum import Enum, auto as enum_auto
import time

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.ros_util import ROSDataConverter
from diffusion_policy.common.pytorch_util import dict_apply

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import rcl_interfaces.msg
import geometry_msgs.msg
import sensor_msgs.msg
import std_msgs.msg
import std_srvs.srv

class ActionType(Enum):
    FORCE = enum_auto()
    POSITION = enum_auto()

class LastMessageSubscriber:
    """Utility class to subscribe a node to topics and store the last received message on those topics."""
    def __init__(
        self,
        action_predictor_node: Node, # actually ActionPredictor
        topic,
        topic_type,
        topic_ndx,
    ):
        """Create a subscription for the input topic."""
        self.action_predictor_node = action_predictor_node
        self.action_predictor_node.create_subscription(
            msg_type=topic_type,
            topic=topic,
            callback=self.callback,
            qos_profile=10
        )
        self.ndx = topic_ndx
    def callback(self, msg):
        """Store the last received message at the correct index and mark it received."""
        self.action_predictor_node.obs_received[self.ndx] = True
        self.action_predictor_node.last_message[self.ndx] = msg


class ActionPredictor(Node):
    """ROS node for predicting actions using Diffusion Policy"""
    def __init__(self):
        """Initialize node and load diffusion policy model."""
        super().__init__('action_predictor')

        # Dictionary to collect model details to publish on the /model_details topic
        # (for keeping track of this while ROS bagging)
        self.model_details = {}
        # Timer for publishing those model details
        self.timer_model_details = self.create_timer(1.0, self.timer_model_details_callback)

        # PARAMETERS
        self.declare_parameter(
            'checkpoint_path',
            '',
            rcl_interfaces.msg.ParameterDescriptor(
                description='Checkpoint file (\'.ckpt\') that contains model weights and config info'
            )
        )
        checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        if checkpoint_path == '':
            raise Exception('No checkpoint provided for loading model')
        self.model_details['checkpoint_path'] = checkpoint_path

        self.declare_parameter(
            'num_inference_diffusion_timesteps',
            16,
            rcl_interfaces.msg.ParameterDescriptor(
                description='Number of timesteps the diffusion model uses for inference.'
                ' Overrides only accepted if a DDIM noise scheduler is used'
            )
        )
        self.num_inference_diffusion_timesteps = self.get_parameter('num_inference_diffusion_timesteps').get_parameter_value().integer_value

        self.declare_parameter(
            'use_residuals',
            False,
            rcl_interfaces.msg.ParameterDescriptor(
                description='If true, residuals are used. That is, the published action is the '
                'predicted action minus the actual current value for that action (from external sources)'
            )
        )
        self.use_residuals = self.get_parameter('use_residuals').get_parameter_value().bool_value

        self.add_on_set_parameters_callback(self.parameters_callback)

        # TODO method of disabling specific cameras

        # PUBLISHERS
        self.pub_action = self.create_publisher(geometry_msgs.msg.Wrench, '/omnid1/delta/additional_force', 10)
        self.pub_model_details = self.create_publisher(std_msgs.msg.String, '/model_details', 10)

        # SUBSCRIBERS
        self.sub_joint_states = self.create_subscription(
            sensor_msgs.msg.JointState,
            '/omnid1/joint/joint_states',
            self.sub_joint_states_callback,
            10
        )

        # SERVICE SERVERS
        self.srv_start_inference = self.create_service(std_srvs.srv.Empty, 'start_inference', self.srv_start_inference_callback)
        self.srv_stop_inference = self.create_service(std_srvs.srv.Empty, 'stop_inference', self.srv_stop_inference_callback)
        self.srv_start_action = self.create_service(std_srvs.srv.Empty, 'start_action', self.srv_start_action_callback)
        self.srv_stop_action = self.create_service(std_srvs.srv.Empty, 'stop_action', self.srv_stop_action_callback)
        self.enable_inference = True
        self.enable_action = False

        # A lot of the code below for loading/using the model is modeled after the
        # eval_real_robot.py script

        # Load payload/config from a checkpoint produced by training script
        self.payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        self.cfg = self.payload['cfg']

        # Determine if the model is lowdim (no images used)
        self.low_dim = 'lowdim' in self.cfg.task.name

        # Determine what the output action is
        if 'output_force' in self.cfg.task.name:
            self.action_type = ActionType.FORCE
        elif 'output_position' in self.cfg.task.name:
            self.action_type = ActionType.POSITION
        else:
            raise Exception('Action type is not implemented in this node')

        # Different models are trained on data that is decimated at different rates
        # this rate is used for the timer that will collect data/perform actions
        self.observation_rate = self.cfg.task.data_conversion.rate
        self.observation_period = 1.0 / self.observation_rate

        # Load workspace (wrapper around model) based on the config
        workspace_cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace: BaseWorkspace = workspace_cls(self.cfg)
        self.workspace.load_payload(self.payload, exclude_keys=None, include_keys=None)

        # Load model
        # BaseImagePolicy type hint is fine even if the policy is only low dim
        self.policy: BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model

        # Enable GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.eval().to(self.device)

        # Set interface parameters
        # DDIM noise schedulers can have a different number of inference diffusion timesteps during 
        # use than during training, so the parameter can be used to set this value
        # DDPM noise schedulers need the same number of inference diffusion timesteps during use
        # as during training, so this value must be set by the config
        if 'DDIM' in self.cfg.policy.noise_scheduler._target_:
            self.policy.num_inference_steps = self.num_inference_diffusion_timesteps
        else:
            self.get_logger().info('DDIM noise scheduler not used, ignoring inference steps parameter')
            self.num_inference_diffusion_timesteps = self.policy.num_inference_steps
            self.set_parameters([Parameter(name='num_inference_diffusion_timesteps', value=self.num_inference_diffusion_timesteps)])
        self.model_details['num_inference_diffusion_timesteps'] = self.num_inference_diffusion_timesteps

        # This can be modified at run time through the parameter callback
        self.declare_parameter(
            'num_actions_taken',
            self.policy.n_action_steps,
            rcl_interfaces.msg.ParameterDescriptor(
                description='Number of actions taken based on an inference before a new inference is used'
            )
        )
        self.num_actions_taken = self.get_parameter('num_actions_taken').get_parameter_value().integer_value
        self.model_details['num_actions_taken'] = self.num_actions_taken

        # This basically just stops the policy from limiting the number of actions in the output
        # of an inference so more actions can be taken if necessary (i.e. if the model inference
        # takes too long)
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
            low_dim_obs_key = 'obs'
        else:
            cameras_to_exclude = [] # TODO add parameter for this
            low_dim_obs_key = 'low_dim_obs'
        # This is a custom data converter that converts ROS message data to the format expected
        # at the input of the model. It is also used by the omnid_bag_conversion.py script to
        # process the training data. How it formats the data is determined by format determined
        # by the data_conversion config file (cfg.task.data_conversion)
        self.data_converter = ROSDataConverter(
            cfg=self.cfg.task.data_conversion,
            exclude_cameras=cameras_to_exclude,
            camera_format='CHW', # for passing into model
            low_dim_obs_key=low_dim_obs_key
        )

        # The data converter determines what topics should be subscribed to
        # topic_info includes stuff like what index in a data frame that topic will be and
        # what the data type of the message is
        topics, topic_info = self.data_converter.get_topics_and_info()

        self.input_data_subscribers = []
        self.last_message = []

        # LastMessageSubscriber is a utility class that handles subscribing to the input topic.
        # In the callback for that topic, it stores the last message received in the proper index of
        # the self.last_message array of this node
        for topic in topics:
            self.last_message.append(None)
            self.input_data_subscribers.append(
                LastMessageSubscriber(self, topic, topic_info[topic]['type'], topic_info[topic]['ndx'])
            )
        self.last_message.append(None) # For timestep, the last element of a message list

        # Timer for storing observations, triggering predictions, and performing actions
        self.timer = self.create_timer(self.observation_period, self.timer_callback)

        # Variables for handling received observations
        self.reset_obs_received()
        self.at_least_one_of_each_obs_received = False
        self.n_obs_received = False
        self.obs_data_queue = []
        self.obs_data_mutex = threading.Lock()

        # Variables for handling inference
        self.inference_counter = self.num_actions_taken
        self.inference_thread = threading.Thread()

        # Variables for handling actions
        self.action_data_mutex = threading.Lock()
        self.action_counter = 0
        self.action_array = []

        self.last_ee_force = geometry_msgs.msg.Vector3()

        self.get_logger().info(f'Subscribing to: {self.data_converter.get_topics()}')
        self.get_logger().info(f'Synchronizing at {self.observation_rate} Hz')
        self.get_logger().info(f'Horizon: {self.policy.horizon}, Observations: {self.policy.n_obs_steps}, Actions: {self.num_actions_taken}')

    def parameters_callback(self, params):
        """Runtime modification of parameters."""
        # TODO implement more parameters if necessary
        success = True
        reason = ''

        for param in params:
            if param.name == 'num_actions_taken':
                if param.value > self.policy.n_action_steps:
                    self.num_actions_taken = self.policy.n_action_steps
                    message = f'Actions cannot be greater than the horizon ({self.policy.n_action_steps}, clamping.'
                    self.get_logger().warn(message)
                    reason += message
                elif param.value < 0:
                    self.num_actions_taken = 0
                    message = 'Actions cannot be less than 0, clamping.'
                    self.get_logger().warn(message)
                    reason += message
                else:
                    self.num_actions_taken = param.value
                self.model_details['num_actions_taken'] = self.num_actions_taken
            elif param.name == 'use_residuals':
                self.use_residuals = param.value

        return rcl_interfaces.msg.SetParametersResult(successful=success, reason=reason)

    def timer_model_details_callback(self):
        """Publish model details regularly for recording in ROS bags."""
        self.model_details['inference_enabled'] = self.enable_inference
        self.model_details['action_enabled'] = self.enable_action
        self.model_details['use_residuals'] = self.use_residuals
        self.pub_model_details.publish(std_msgs.msg.String(data=yaml.dump(self.model_details)))

    def sub_joint_states_callback(self, msg: sensor_msgs.msg.JointState):
        """Get last end effector forces for use with residual force actions."""
        x_ndx = msg.name.index('x')
        y_ndx = msg.name.index('y')
        z_ndx = msg.name.index('z')

        self.last_ee_force.x = msg.effort[x_ndx]
        self.last_ee_force.y = msg.effort[y_ndx]
        self.last_ee_force.z = msg.effort[z_ndx]

    def reset_obs_received(self):
        """Reset array which keeps track of whether a message has been received on all observation topics."""
        self.obs_received = np.full((len(self.last_message) - 1,), False)

    def srv_start_inference_callback(self, request, response):
        """Service to enable infering actions using the diffusion policy model."""
        self.enable_inference = True
        self.inference_counter = self.num_actions_taken  # immediately trigger inference
        return response

    def srv_stop_inference_callback(self, request, response):
        """Service to disable infering actions using the diffusion policy model. Also disables performing actions."""
        self.enable_inference = False
        self.srv_stop_action_callback(None, None)
        return response

    def srv_start_action_callback(self, request, response):
        """Service to enable performing actions using the inferred actions."""
        if self.enable_inference:
            self.enable_action = True
        return response
    
    def srv_stop_action_callback(self, request, response):
        """Service to disable performing actions using the inferred actions."""
        self.enable_action = False
        self.action_array = []
        self.action_counter = 0
        return response

    def timer_callback(self):
        """Main timer callback that handles recording observations, triggering inferences, and performing actions."""
        
        # Check that an observation has been received on all input topics
        if np.any(~self.obs_received):
            if not self.at_least_one_of_each_obs_received:
                # we haven't received one of each observation yet, no point in continuing
                return

            # Warn user that an observation was not received on a particular topic.
            # in this case, a previous value (the last stored) will be used.
            for i in range(len(self.obs_received)):
                if not self.obs_received[i]:
                    self.get_logger().warn(f'{self.data_converter.get_topics()[i]} not received in the last observation period, using previous value')
        self.at_least_one_of_each_obs_received = True
        self.reset_obs_received()


        # Lock mutex to modify observation data queue
        with self.obs_data_mutex:
            # set timestep for observation based on current clock
            # honestly this "synchronization" is pretty terrible and isn't really synchronization at
            # all. But I didn't like how the message_filters library was performing (I couldn't
            # control the rate) and didn't have a ton of time to make a better solution
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

        # This counts timer iterations between inferences, allowing inferences to be triggered
        # at the proper time to complete so only self.num_actions_taken actions are taken
        # from an action inference before a new inference is available
        self.inference_counter += 1

        # Only start a new inference if the last inference has completed
        if not self.inference_thread.is_alive():
            # If it's time to infer again, start thread
            if self.inference_counter > self.num_actions_taken:
                self.inference_counter = 0
                self.inference_thread = threading.Thread(target=self.infer)
                self.inference_thread.start()

        # Perform actions from past inferences
        # Mutex is used to protect inferred action data from simultaneous access by multiple threads
        with self.action_data_mutex:

            # Perform actions until the end of the inferred action list is reached
            if self.action_counter < len(self.action_array):
                x = float(self.action_array[self.action_counter][0])
                y = float(self.action_array[self.action_counter][1])

                if self.use_residuals:
                    x -= self.last_ee_force.x
                    y -= self.last_ee_force.y

                if self.enable_action:
                    msg = geometry_msgs.msg.Wrench()
                    msg.force.x = x
                    msg.force.y = y
                    self.pub_action.publish(msg)
                    print(f'Performing action {self.action_counter}: {x, y}')
                else:
                    print(f'Simulating action {self.action_counter}: {x, y}')
                
                # Keeps track of which action in the inferred action list is being performed.
                # is reset when a new inference completes.
                self.action_counter += 1


    def infer(self):
        """Method that performs the inference in a parallel thread so as to not block collecting observations/performing actions."""

        self.get_logger().info('Starting inference')
        start = time.time()

        # Convert observation data from ROS messages to np arrays for input into model
        # Lock mutex to read observation data queue
        with self.obs_data_mutex:
            obs_data = self.data_converter.convert_data_frames(self.obs_data_queue)

        # remove unnecessary data so it's not converted into a Tensor
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

        # Replace previous inferred action list and reset performed action counter
        # Mutex is used to protect inferred action data from simultaneous access by multiple threads
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