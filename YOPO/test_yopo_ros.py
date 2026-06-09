import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from threading import Lock
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2

import cv2
import os
import time
import torch
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from config.config import cfg
from control_msg import PositionCommand
from policy.yopo_network import YopoNetwork
from policy.poly_solver import *
from policy.state_transform import *

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class YopoNet:
    def __init__(self, config, weight):
        self.config = config
        rospy.init_node('yopo_net', anonymous=False)
        # load params
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.min_dis, self.max_dis = 0.4, 6.0
        self.goal = np.array(self.config['goal'])
        self.plan_from_reference = self.config['plan_from_reference']
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.show_depth = self.config.get('show_depth', False)
        self.publish_processed_depth = self.config.get('publish_processed_depth', False)
        self.depth_vis_topic = self.config.get('depth_vis_topic', '/yopo_net/processed_depth')
        self.depth_window_name = self.config.get('depth_window_name', 'YOPO Processed Depth')
        self.depth_window_initialized = False
        self.depth_colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix() # 相机到机体的旋转矩阵,相机俯仰角
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # variables
        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.depth_fps = 30  # used only as processing time tolerance for printing logs

        # Load Network
        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            state_dict = torch.load(weight, weights_only=True)
            self.policy = YopoNetwork() # 网络层结构,包含forward方法
            self.policy.load_state_dict(state_dict) # 加载权重参数
            self.policy = self.policy.to(self.device) # 将模型加载到GPU或CPU
            self.policy.eval() # 设置为推理模式，关闭dropout和batchnorm等训练特有的操作
        self.warm_up() # 模拟一个全0的深度图和观测输入进行前向推理，预热GPU

        # ros publisher
        self.lattice_traj_pub = rospy.Publisher("/yopo_net/lattice_trajs_visual", PointCloud2, queue_size=1) # 发布可视化候选轨迹点云
        self.best_traj_pub = rospy.Publisher("/yopo_net/best_traj_visual", PointCloud2, queue_size=1) # 发布最优轨迹点云
        self.all_trajs_pub = rospy.Publisher("/yopo_net/trajs_visual", PointCloud2, queue_size=1) # 可视化所有轨迹候选
        self.processed_depth_pub = rospy.Publisher(self.depth_vis_topic, Image, queue_size=1) if self.publish_processed_depth else None
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1) # 发布无人机控制指令
        # ros subscriber
        self.odom_sub = rospy.Subscriber(self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True) # 订阅里程计
        self.depth_sub = rospy.Subscriber(self.config['depth_topic'], Image, self.callback_depth, queue_size=1, tcp_nodelay=True) # 订阅深度图像
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1) # 获取设定目标点
        # ros timer
        rospy.on_shutdown(self.shutdown_hook)
        rospy.sleep(1.0)  # wait connection...
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub) # 定时执行无人机控制逻辑并发布
        print("YOPO Net Node Ready!")
        rospy.spin() # 阻塞当前线程并让节点持续运行

    def callback_set_goal(self, data):
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, data.pose.position.z]) # 固定高度2米
        self.arrive = False
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f}, {data.pose.position.z:.1f})")

    # the first frame
    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init: # 第一次执行初始化,期望位置为当前位置
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        if np.linalg.norm(pos - self.goal) < 1.0 and not self.arrive:
            print("Arrive!")
            self.arrive = True

    def process_odom(self):
        # Rwb -> Rwc -> Rcw
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        vel_w = self.desire_vel if self.plan_from_reference else np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(Rotation_cw, acc_w)

        # goal_dir
        goal_w = self.goal - self.desire_pos
        goal_c = np.dot(Rotation_cw, goal_w)

        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm

    @torch.inference_mode()
    def callback_depth(self, data):
        if not self.odom_init: return # 里程计初始化后再进行下面推理

        # 1. Depth Image Process (Be careful with the depth units in your application)
        time0 = time.time()
        if data.encoding == "32FC1":    # Simulator, meter 测距单位米
            depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
        elif data.encoding == "16UC1":  # RealSense, millimeter 测距单位毫米
            depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width).astype(np.float32) / 1000.0
        else:
            raise ValueError(f"Unsupported depth encoding: {data.encoding}. Expected '32FC1' or '16UC1'.")

        # 调整深度图到网络要求的大小
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # 剪裁过远的距离并归一化 [0,1]
        depth = np.minimum(depth, self.max_dis) / self.max_dis

        # interpolated the nan value (experiment shows that treating nan directly as 0 produces similar results)
        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis) # 检测深度图中 NaN 值 和过近的值
        interpolated_image = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS) # 填补图像中无效区域
        interpolated_image = interpolated_image.astype(np.float32) / 255.0 # 再次归一化到 [0, 1]
        self.visualize_depth(interpolated_image, data.header)
        depth = interpolated_image.reshape([1, 1, self.height, self.width])

        # 2. YOPO Network Inference
        # input prepare
        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)  # (non_blocking: copying speed 3x) 将深度图从CPU传输到GPU
        obs_norm = self.process_odom().to(self.device, non_blocking=True) # 将观测状态从CPU传输到GPU
        obs_input = self.state_transform.prepare_input(obs_norm) # 坐标变换到基元坐标系下,准备CNN输入格式
        # torch.cuda.synchronize()

        time2 = time.time()
        # Forward (TensorRT: inference speed increased by 5x)
        endstate_pred, score_pred = self.policy(depth_input, obs_input) # 前向推理
        endstate_pred, score_pred = endstate_pred.cpu().numpy(), score_pred.cpu().numpy() # 将结果从GPU传输到CPU
        time3 = time.time()

        # 3. Post-Processing
        # Replacing PyTorch operation on CUDA with NumPy operation on CPU (speed increased by 10x)
        endstate, score = self.process_output(endstate_pred, score_pred, return_all_preds=self.visualize) # 将预测的 primitive frame 末态转换到 body frame
        # Vectorization: transform the prediction(P V A in body frame) to the world frame with the attitude (without the position)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)  # [N, 9] -> [N, 3, 3] -> [px vx ax, py vy ay, pz vz az]
        endstate_w = np.matmul(self.Rotation_wc, endstate_c) # 将末态从 body frame → 世界坐标系

        action_id = np.argmin(score) if self.visualize else 0 # 根据 score 选择最优 primitive → action_id
        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety
            # 使用当前里程计位置速度作为起点
            start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0], endstate_w[action_id, 0, 0] + start_pos[0],
                                              endstate_w[action_id, 0, 1], endstate_w[action_id, 0, 2], self.traj_time)
            self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1], endstate_w[action_id, 1, 0] + start_pos[1],
                                              endstate_w[action_id, 1, 1], endstate_w[action_id, 1, 2], self.traj_time)
            self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2], endstate_w[action_id, 2, 0] + start_pos[2],
                                              endstate_w[action_id, 2, 1], endstate_w[action_id, 2, 2], self.traj_time)
            self.ctrl_time = 0.0
        time4 = time.time()
        self.visualize_trajectory(score_pred, endstate_w)
        time5 = time.time()

        self.print_time(time0, time1, time2, time3, time4, time5)

    def control_pub(self, _timer):
        if self.ctrl_time is None or self.ctrl_time > self.traj_time: # 如果控制时间未初始化或者轨迹已经结束（ctrl_time > traj_time），直接返回
            return
        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False   # ready for next rollout
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
            return

        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety and publish frequency
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY

            # 沿轨迹计算状态（位置、速度、加速度）
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)

            # 更新期望状态
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])

            goal_dir = self.goal - self.desire_pos # 当前位置到目标的方向向量
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt) # 计算无人机航向角和航向角速度
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)

    def visualize_depth(self, depth_image, header=None):
        if not self.show_depth and self.processed_depth_pub is None:
            return

        # 处理后的深度图仍是 [0, 1]，这里转成更容易看懂的伪彩色图。
        depth_uint8 = np.clip(depth_image * 255.0, 0, 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(255 - depth_uint8, self.depth_colormap)

        if self.show_depth:
            try:
                if not self.depth_window_initialized:
                    cv2.namedWindow(self.depth_window_name, cv2.WINDOW_NORMAL)
                    self.depth_window_initialized = True
                cv2.imshow(self.depth_window_name, depth_vis)
                cv2.waitKey(1)
            except cv2.error as exc:
                rospy.logwarn_throttle(5.0, f"OpenCV depth preview disabled: {exc}")
                self.show_depth = False

        if self.processed_depth_pub is not None and self.processed_depth_pub.get_num_connections() > 0:
            vis_msg = Image()
            vis_msg.header = header if header is not None else std_msgs.msg.Header(stamp=rospy.Time.now())
            vis_msg.height, vis_msg.width = depth_vis.shape[:2]
            vis_msg.encoding = "bgr8"
            vis_msg.is_bigendian = 0
            vis_msg.step = depth_vis.shape[1] * depth_vis.shape[2]
            vis_msg.data = depth_vis.tobytes()
            self.processed_depth_pub.publish(vis_msg)

    def shutdown_hook(self):
        if self.depth_window_initialized:
            try:
                cv2.destroyWindow(self.depth_window_name)
            except cv2.error:
                pass

    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)

        if not return_all_preds:
            action_id = np.argmin(score_pred)
            lattice_id = self.lattice_primitive.traj_num - 1 - action_id
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred[action_id, :][np.newaxis, :], lattice_id)
            score = score_pred[action_id]
        else:
            score = score_pred
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, torch.arange(self.lattice_primitive.traj_num-1, -1, -1))

        return endstate, score

    def visualize_trajectory(self, pred_score, pred_endstate):
        dt = self.traj_time / 20.0
        start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        # best predicted trajectory
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                self.optimal_poly_x.get_position(t_values),
                self.optimal_poly_y.get_position(t_values),
                self.optimal_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.best_traj_pub.publish(point_cloud_msg)
        # lattice primitive
        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = np.dot(lattice_endstate, self.Rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            lattice_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                          lattice_endstate[:, 0] + start_pos[0], zero_state[:, 0], zero_state[:, 0], self.traj_time)
            lattice_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                          lattice_endstate[:, 1] + start_pos[1], zero_state[:, 1], zero_state[:, 1], self.traj_time)
            lattice_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                          lattice_endstate[:, 2] + start_pos[2], zero_state[:, 2], zero_state[:, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                lattice_poly_x.get_position(t_values),
                lattice_poly_y.get_position(t_values),
                lattice_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.lattice_traj_pub.publish(point_cloud_msg)
        # all predicted trajectories
        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            all_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                      pred_endstate[:, 0, 0] + start_pos[0], pred_endstate[:, 0, 1], pred_endstate[:, 0, 2], self.traj_time)
            all_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                      pred_endstate[:, 1, 0] + start_pos[1], pred_endstate[:, 1, 1], pred_endstate[:, 1, 2], self.traj_time)
            all_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                      pred_endstate[:, 2, 0] + start_pos[2], pred_endstate[:, 2, 1], pred_endstate[:, 2, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                all_poly_x.get_position(t_values),
                all_poly_y.get_position(t_values),
                all_poly_z.get_position(t_values)
            ), axis=-1)
            scores = np.repeat(pred_score, t_values.size)
            points_array = np.column_stack((points_array, scores))
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.FLOAT32, 1)]
            point_cloud_msg = point_cloud2.create_cloud(header, fields, points_array)
            self.all_trajs_pub.publish(point_cloud_msg)

    def print_time(self, time0, time1, time2, time3, time4, time5):
        """
        Performance reference: PyTorch model should take < 5 ms; TensorRT model should take < 1 ms

        Notes:
        - Running program and enabling RViz under WSL greatly increase processing time, and Ubuntu does not have these issues
        - Even with queue_size=1, it may cause message accumulation and lag when processing time exceeds the image frequency
        """
        self.time_interpolation = self.time_interpolation + (time1 - time0)
        self.time_prepare = self.time_prepare + (time2 - time1)
        self.time_forward = self.time_forward + (time3 - time2)
        self.time_process = self.time_process + (time4 - time3)
        self.time_visualize = self.time_visualize + (time5 - time4)
        self.count = self.count + 1

        total_time = (time5 - time0) * 1000
        tolerance = 1000.0 / self.depth_fps
        if total_time > tolerance:
            rospy.logwarn(f"Warn: Processing time {(time5 - time0) * 1000:.2f} ms exceeds {tolerance:.2f} ms, may cause message lag!")
            print(f"\033[34mCurrent Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * (time1 - time0):.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * (time2 - time1):.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * (time3 - time2):.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * (time4 - time3):.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * (time5 - time4):.2f} ms\033[0m")
        if self.verbose or (total_time > tolerance):
            print(f"\033[34mAverage Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * self.time_interpolation / self.count:.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * self.time_prepare / self.count:.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * self.time_forward / self.count:.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * self.time_process / self.count:.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * self.time_visualize / self.count:.2f} ms\033[0m")

    def warm_up(self):
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device) # 模拟一个全0的深度图输入
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device) # 模拟一个全0的观测输入
        obs = self.state_transform.prepare_input(obs) # 转换成BCVH格式
        endstate_pred, score_pred = self.policy(depth, obs) # 前向推理
        _ = self.state_transform.pred_to_endstate(endstate_pred)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    return parser


if __name__ == "__main__":

    # 解析外部的传入参数 --trial=1 --epoch=50
    args = parser().parse_args()

    # 获取当前绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载权重文件 使用tensorRT yopo_trt.pth 或 不使用tensorRT YOPO_{trial}/epoch{epoch}.pth
    weight = "yopo_trt.pth" if args.use_tensorrt else base_dir + "/saved/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch)
    print("load weight from:", weight)

    # 配置参数
    settings = {'use_tensorrt': args.use_tensorrt,
                'goal': [50, 0, 2],      # 目标点位置
                'pitch_angle_deg': -0,   # 相机俯仰角(仰为负)
                'odom_topic': '/sim/odom',                   # 里程计话题
                'depth_topic': '/depth_image',               # 深度图话题
                'ctrl_topic': '/so3_control/pos_cmd',        # 控制器话题
                'plan_from_reference': False,   # 从参考状态规划
                'verbose': False,               # 打印耗时？
                'visualize': True,              # 可视化所有轨迹？(实飞改为False节省计算)
                'show_depth': False,             # 本机弹窗显示处理后的深度图
                'publish_processed_depth': True,  # 发布处理后的深度图，便于 rqt_image_view / RViz 查看
                'depth_vis_topic': '/yopo_net/processed_depth'
                }
    YopoNet(settings, weight)
