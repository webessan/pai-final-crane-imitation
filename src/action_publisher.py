#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time

class RobotActionPublisher(Node):
    def __init__(self):
        super().__init__('robot_action_publisher')
        
        # 1. アーム制御用パブリッシャー
        self.arm_publisher = self.create_publisher(
            JointTrajectory,
            'crane_plus_arm_controller/joint_trajectory',
            10)
            
        # 2. グリッパ（手先）制御用パブリッシャーを追加
        self.gripper_publisher = self.create_publisher(
            JointTrajectory,
            'crane_plus_gripper_controller/joint_trajectory',
            10)
            
        self.get_logger().info('Robot Action Publisher node has been started')

    # アーム（4関節）を動かす関数
    def move_to_position(self, positions, time_from_start=2.0):
        msg = JointTrajectory()
        msg.joint_names = [
            'crane_plus_joint1',
            'crane_plus_joint2',
            'crane_plus_joint3',
            'crane_plus_joint4'
        ]
        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * len(positions)
        point.accelerations = [0.0] * len(positions)
        point.time_from_start = Duration(
            sec=int(time_from_start),
            nanosec=int((time_from_start % 1) * 1e9)
        )
        msg.points.append(point)
        self.arm_publisher.publish(msg)

    # グリッパ（手先）を動かす関数を追加
    def move_gripper(self, position, time_from_start=2.0):
        msg = JointTrajectory()
        msg.joint_names = ['crane_plus_joint_hand']
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.velocities = [0.0]
        point.accelerations = [0.0]
        point.time_from_start = Duration(
            sec=int(time_from_start),
            nanosec=int((time_from_start % 1) * 1e9)
        )
        msg.points.append(point)
        self.gripper_publisher.publish(msg)


def main():
    rclpy.init()
    robot_action_publisher = RobotActionPublisher()
    time.sleep(1.0)

    try:
        # 1. 初期姿勢 & グリッパを開く (0.0 rad)
        robot_action_publisher.move_to_position([0.0, 0.0, 0.0, 0.0])
        robot_action_publisher.move_gripper(-0.5236)
        time.sleep(3.0)

        # 2. キューブの上空へ移動（衝突防止）
        robot_action_publisher.move_to_position([1.60, 0.0, 1.0, 1.0])
        time.sleep(3.0)

        # 3. キューブを挟む位置へ下りる
        robot_action_publisher.move_to_position([1.60, 0.10, 1.75, 1.35])
        time.sleep(3.0)

        # 4. グリッパを閉じる（メモした角度 0.15 rad で掴む）
        robot_action_publisher.move_gripper(0.35)
        time.sleep(3.0)

        # 5. 初期姿勢に戻る（キューブを持ち上げる）
        robot_action_publisher.move_to_position([0.0, 0.0, 0.0, 0.0])
        time.sleep(3.0)

        robot_action_publisher.get_logger().info('All movements completed. Exiting...')

    except KeyboardInterrupt:
        robot_action_publisher.get_logger().info('Movement interrupted by user')
    finally:
        robot_action_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()