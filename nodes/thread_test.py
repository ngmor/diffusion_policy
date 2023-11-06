#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from threading import Thread, Lock

class ThreadTest(Node):
    def __init__(self):
        super().__init__('thread_test')

        self.n_actions = 8
        self.horizon = 16

        self.inference_counter = self.n_actions
        self.action_counter = 0
        self.action_list = []



        self.timer = self.create_timer(0.5, self.timer_callback)

        self.thread = Thread(target=self.thread_function)
        self.mutex = Lock()

    def timer_callback(self):
        # Collect observation data here

        # Count up number of callbacks since last inference
        self.inference_counter += 1

        if not self.thread.is_alive(): # Thread is running
            # If it's time to infer again, start thread
            if self.inference_counter > self.n_actions:
                print('Starting inference')
                self.inference_counter = 0
                self.thread = Thread(target=self.thread_function)
                self.thread.start()

        # Perform action
        with self.mutex:
            if self.action_counter < len(self.action_list):
                print(f'Performing action: {self.action_list[self.action_counter]}')
                self.action_counter += 1

    def thread_function(self):
        # Simulate a long running calculationg
        import time
        time.sleep(2)

        with self.mutex:
            self.action_list = []
            for i in range(self.horizon):
                self.action_list.append(i)
            self.action_counter = 0
            print('Inference complete')


def main(args=None):
    rclpy.init(args=args)
    node = ThreadTest()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()