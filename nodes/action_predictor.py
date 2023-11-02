#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class ActionPredictor(Node):
    def __init__(self):
        super().__init__('action_predictor')

def main(args=None):
    rclpy.init(args=args)
    node = ActionPredictor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()