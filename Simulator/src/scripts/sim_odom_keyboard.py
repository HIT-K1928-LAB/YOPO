#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import select
import sys
import termios
import tty

import rospy
from geometry_msgs.msg import Point, Quaternion
from nav_msgs.msg import Odometry
import tf


HELP = """
Keyboard odom simulator
-----------------------
w / s : forward / backward
a / d : turn left / turn right
r / f : rise / fall
space : hover
q     : quit

Hold a key to keep moving. Releasing the key will stop motion after a short timeout.
"""


class KeyboardOdomPublisher(object):
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/sim/odom")
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.child_frame_id = rospy.get_param("~child_frame_id", "base_link")
        self.publish_rate = float(rospy.get_param("~publish_rate", 30.0))
        self.linear_speed = float(rospy.get_param("~linear_speed", 1.0))
        self.vertical_speed = float(rospy.get_param("~vertical_speed", 0.8))
        self.yaw_rate = float(rospy.get_param("~yaw_rate", 0.8))
        self.command_timeout = float(rospy.get_param("~command_timeout", 0.2))

        self.x = float(rospy.get_param("~start_x", 0.0))
        self.y = float(rospy.get_param("~start_y", 2.0))
        self.z = float(rospy.get_param("~start_z", 1.6))
        self.yaw = float(rospy.get_param("~start_yaw", 0.0))

        self.forward_velocity = 0.0
        self.vertical_velocity = 0.0
        self.angular_velocity = 0.0

        self.pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)
        self.last_update_time = rospy.Time.now()
        self.last_command_time = rospy.Time.now()
        self.terminal_settings = None

        if not sys.stdin.isatty():
            raise RuntimeError("Keyboard control requires a terminal (TTY). Please run with rosrun in a shell.")

        self.stdin_fd = sys.stdin.fileno()
        self.terminal_settings = termios.tcgetattr(self.stdin_fd)
        tty.setcbreak(self.stdin_fd)
        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("Publishing keyboard odometry on %s", self.odom_topic)
        rospy.loginfo(HELP)

    def on_shutdown(self):
        self.stop_motion()
        if self.terminal_settings is not None:
            termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.terminal_settings)

    def stop_motion(self):
        self.forward_velocity = 0.0
        self.vertical_velocity = 0.0
        self.angular_velocity = 0.0

    def get_key(self):
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        key = sys.stdin.read(1) if readable else ""
        return key

    def handle_key(self, key):
        if key == "w":
            self.forward_velocity = self.linear_speed
            self.vertical_velocity = 0.0
            self.angular_velocity = 0.0
        elif key == "s":
            self.forward_velocity = -self.linear_speed
            self.vertical_velocity = 0.0
            self.angular_velocity = 0.0
        elif key == "a":
            self.forward_velocity = 0.0
            self.vertical_velocity = 0.0
            self.angular_velocity = self.yaw_rate
        elif key == "d":
            self.forward_velocity = 0.0
            self.vertical_velocity = 0.0
            self.angular_velocity = -self.yaw_rate
        elif key == "r":
            self.forward_velocity = 0.0
            self.vertical_velocity = self.vertical_speed
            self.angular_velocity = 0.0
        elif key == "f":
            self.forward_velocity = 0.0
            self.vertical_velocity = -self.vertical_speed
            self.angular_velocity = 0.0
        elif key == " ":
            self.stop_motion()
        elif key == "q":
            rospy.signal_shutdown("User requested exit.")
            return
        else:
            return

        self.last_command_time = rospy.Time.now()

    def update_pose(self, now):
        dt = (now - self.last_update_time).to_sec()
        if dt <= 0.0:
            return

        if (now - self.last_command_time).to_sec() > self.command_timeout:
            self.stop_motion()

        self.yaw += self.angular_velocity * dt
        world_vx = self.forward_velocity * math.cos(self.yaw)
        world_vy = self.forward_velocity * math.sin(self.yaw)

        self.x += world_vx * dt
        self.y += world_vy * dt
        self.z = max(0.0, self.z + self.vertical_velocity * dt)
        self.last_update_time = now

    def publish_odom(self, now):
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = self.frame_id
        odom_msg.child_frame_id = self.child_frame_id
        odom_msg.pose.pose.position = Point(self.x, self.y, self.z)

        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, self.yaw)
        odom_msg.pose.pose.orientation = Quaternion(*quaternion)

        odom_msg.twist.twist.linear.x = self.forward_velocity
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = self.vertical_velocity
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = self.angular_velocity

        self.pub.publish(odom_msg)

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            self.update_pose(now)

            key = self.get_key()
            if key:
                self.handle_key(key)

            self.publish_odom(rospy.Time.now())
            rate.sleep()


def main():
    rospy.init_node("sim_odom_keyboard")
    publisher = KeyboardOdomPublisher()
    publisher.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except RuntimeError as exc:
        sys.stderr.write(str(exc) + "\n")
