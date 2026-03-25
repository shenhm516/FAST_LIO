#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Point Cloud Registration ROS Node using ICP Algorithm
"""

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from collections import deque
from typing import Tuple, Optional
import os
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation


def icp_2d(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    max_distance: float = 0.5,
    init_x: float = 0.0,
    init_y: float = 0.0,
    init_yaw: float = 0.0,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    手写2D ICP点云配准算法

    Args:
        source: 源点云 (N, 2)
        target: 目标点云 (M, 2)
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
        max_distance: 最近邻搜索的最大距离
        init_x: 初始x平移
        init_y: 初始y平移
        init_yaw: 初始yaw角度（弧度）
        verbose: 是否打印迭代信息

    Returns:
        transformation: 4x4变换矩阵
        final_error: 最终误差
    """
    # 构建初始变换
    R_init = Rotation.from_euler('z', init_yaw).as_matrix()
    t_init = np.array([init_x, init_y])

    # 应用初始变换到源点云
    current_source = (R_init[:2,:2] @ source.T).T + t_init

    # 累积变换
    R_total = R_init[:2,:2]
    t_total = t_init.copy()

    # 构建目标点云的KDTree
    target_tree = KDTree(target)

    prev_error = float('inf')

    if verbose:
        rospy.loginfo(f"Initial pose: x={init_x:.4f}, y={init_y:.4f}, yaw={np.degrees(init_yaw):.2f}°")
        rospy.loginfo("Running 2D ICP...")

    for iteration in range(max_iterations):
        # 1. 找最近点对应
        distances, indices = target_tree.query(current_source)

        # 过滤距离过大的点
        valid_mask = distances < max_distance
        if np.sum(valid_mask) < 3:
            if verbose:
                rospy.loginfo(f"Iter {iteration}: too few correspondences ({np.sum(valid_mask)}), stop")
            break

        source_matched = current_source[valid_mask]
        target_matched = target[indices[valid_mask]]

        # 2. 计算当前误差
        error = np.sqrt(np.mean(distances[valid_mask] ** 2))

        # 3. 检查收敛
        if abs(prev_error - error) < tolerance:
            if verbose:
                rospy.loginfo(f"Iter {iteration}: converged, error={error:.6f}")
            break
        prev_error = error

        # 4. SVD计算最优变换
        # 计算质心
        centroid_s = np.mean(source_matched, axis=0)
        centroid_t = np.mean(target_matched, axis=0)

        # 去中心化
        source_centered = source_matched - centroid_s
        target_centered = target_matched - centroid_t

        # 计算协方差矩阵
        H = source_centered.T @ target_centered

        # SVD分解
        U, _, Vt = np.linalg.svd(H)
        R_delta = Vt.T @ U.T

        # 确保是纯旋转（行列式为1）
        if np.linalg.det(R_delta) < 0:
            Vt[1, :] *= -1
            R_delta = Vt.T @ U.T

        # 计算平移
        t_delta = centroid_t - R_delta @ centroid_s

        # 5. 更新变换
        current_source = (R_delta @ current_source.T).T + t_delta

        # 累积变换: T_new = T_delta * T_old
        R_total = R_delta @ R_total
        t_total = R_delta @ t_total + t_delta

        if verbose and iteration % 10 == 0:
            yaw_total = np.arctan2(R_total[1, 0], R_total[0, 0])
            rospy.loginfo(f"Iter {iteration}: error={error:.6f}, yaw={np.degrees(yaw_total):.2f}°, "
                          f"t=[{t_total[0]:.4f}, {t_total[1]:.4f}]")

    # 构建4x4变换矩阵
    transformation = np.eye(4)
    transformation[:2, :2] = R_total
    transformation[:2, 3] = t_total

    # 计算最终误差
    distances, _ = target_tree.query(current_source)
    final_error = np.sqrt(np.mean(distances ** 2))

    if verbose:
        yaw_final = np.arctan2(R_total[1, 0], R_total[0, 0])
        rospy.loginfo(f"ICP completed: final_error={final_error:.6f}")
        rospy.loginfo(f"  Yaw: {np.degrees(yaw_final):.2f}°")
        rospy.loginfo(f"  Translation: [{t_total[0]:.4f}, {t_total[1]:.4f}]")

    return transformation, final_error


def extract_2d_slice(
    points_3d: np.ndarray,
    z_center: float = 1.5,
    z_tolerance: float = 0.1,
    max_points: Optional[int] = None
) -> np.ndarray:
    """
    从3D点云中提取z轴在指定范围内的2D点云

    Args:
        points_3d: 3D点云数组 (N, 3)
        z_center: z轴中心高度（米）
        z_tolerance: z轴容忍范围（米）
        max_points: 最大采样点数（None表示不限制）

    Returns:
        2D点云数组 (N, 2)，包含x, y坐标
    """
    z_min = z_center - z_tolerance
    z_max = z_center + z_tolerance
    mask = (points_3d[:, 2] >= z_min) & (points_3d[:, 2] <= z_max)
    points_filtered = points_3d[mask]

    # 提取2D点云 (x, y)
    points_2d = points_filtered[:, :2]

    # 可选：随机采样
    if max_points is not None and len(points_2d) > max_points:
        indices = np.random.choice(len(points_2d), max_points, replace=False)
        points_2d = points_2d[indices]

    return points_2d


def transform_3d_points(
    points: np.ndarray,
    transformation: np.ndarray
) -> np.ndarray:
    """
    对3D点云应用4x4变换矩阵

    Args:
        points: 输入点云 (N, 3)
        transformation: 4x4变换矩阵

    Returns:
        变换后的点云 (N, 3)
    """
    # 应用变换: R @ p + t
    transformed = (transformation[:3, :3] @ points.T).T + transformation[:3, 3]
    return transformed


def create_point_cloud_msg(
    points: np.ndarray,
    frame_id: str = "map"
) -> PointCloud2:
    """
    将点云转换为PointCloud2消息

    Args:
        points: 点云 (N, 3)
        frame_id: 坐标系ID

    Returns:
        PointCloud2消息
    """
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
        pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
        pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1)
    ]

    cloud_msg = pc2.create_cloud(header, fields, points.astype(np.float32))
    return cloud_msg


class ICP2DRegistrationNode:
    """
    ROS节点：订阅点云并进行2D ICP配准
    """

    def __init__(self):
        rospy.init_node('icp_2d_registration_node', anonymous=True)

        # 参数
        self.z_center = rospy.get_param('~z_center', 1.5)
        self.z_tolerance = rospy.get_param('~z_tolerance', 0.1)
        self.max_points = rospy.get_param('~max_points', None)
        self.max_iterations = rospy.get_param('~max_iterations', 100)
        # self.tolerance = rospy.get_param('~tolerance', 1e-6)
        self.max_distance = rospy.get_param('~max_distance', 0.5)

        # 滑窗参数
        self.window_size = rospy.get_param('~window_size', 10)
        self.cloud_window = deque(maxlen=self.window_size)  # 存储最近的N帧2D点云

        # 初值参数
        self.init_x = rospy.get_param('~init_x', 0.0)
        self.init_y = rospy.get_param('~init_y', 0.0)
        self.init_yaw = rospy.get_param('~init_yaw', 0.0)  # 弧度

        # 读取PCD文件作为源点云
        self.source_cloud_2d = None
        self.source_cloud_3d = None
        pcd_path = rospy.get_param('~pcd_path', '/mnt/nas/dataset/rosbag/UniLPR/AEROMAZE/house1.pcd')
        if os.path.exists(pcd_path):
            rospy.loginfo(f"Loading PCD file: {pcd_path}")
            pcd = o3d.io.read_point_cloud(pcd_path)
            points_3d = np.asarray(pcd.points)
            rospy.loginfo(f"Loaded {len(points_3d)} points from PCD")
            self.source_cloud_3d = points_3d
            self.source_cloud_2d = extract_2d_slice(
                points_3d,
                z_center=self.z_center,
                z_tolerance=self.z_tolerance,
                max_points=self.max_points
            )
            rospy.loginfo(f"Source cloud loaded: {len(self.source_cloud_2d)} 2D points for ICP, {len(self.source_cloud_3d)} 3D points for output")
        else:
            rospy.logwarn(f"PCD file not found: {pcd_path}")

        # 订阅点云话题
        self.subscriber = rospy.Subscriber('cloud_registered', PointCloud2, self.cloud_callback, queue_size=1)

        # 发布变换后的点云
        self.transformed_cloud_pub = rospy.Publisher('cloud_house', PointCloud2, queue_size=1)

        rospy.loginfo(f"ICP 2D Registration Node initialized")
        rospy.loginfo(f"z_center: {self.z_center}m, z_tolerance: {self.z_tolerance}m")
        rospy.loginfo(f"Sliding window size: {self.window_size}")
        rospy.loginfo(f"Initial pose: x={self.init_x}, y={self.init_y}, yaw={np.degrees(self.init_yaw):.2f}°")

    def cloud_callback(self, msg: PointCloud2):
        """
        点云回调函数

        Args:
            msg: PointCloud2消息
        """
        rospy.loginfo(f"Received point cloud with {msg.width * msg.height} points")

        # 将PointCloud2转换为numpy数组
        points_3d = []
        for point in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            points_3d.append([point[0], point[1], point[2]])

        if len(points_3d) < 100:
            rospy.logwarn("Too few points received, skipping...")
            return

        points_3d = np.array(points_3d)
        rospy.loginfo(f"Converted to numpy array: {points_3d.shape}")

        # 提取2D点云
        points_2d = extract_2d_slice(
            points_3d,
            z_center=self.z_center,
            z_tolerance=self.z_tolerance,
            max_points=self.max_points
        )
        rospy.loginfo(f"Extracted 2D slice: {len(points_2d)} points (z: {self.z_center}±{self.z_tolerance}m)")

        if len(points_2d) < 10:
            rospy.logwarn("Too few points in 2D slice, skipping...")
            return

        # 将当前帧加入滑窗
        self.cloud_window.append(points_2d)
        rospy.loginfo(f"Added to sliding window, current size: {len(self.cloud_window)}/{self.window_size}")

        # 如果滑窗未满，等待更多帧
        if len(self.cloud_window) < self.window_size:
            rospy.loginfo(f"Waiting for more frames ({len(self.cloud_window)}/{self.window_size})")
            return

        # 合并滑窗中的所有点云作为目标点云
        target_cloud = np.vstack(list(self.cloud_window))
        rospy.loginfo(f"Combined target cloud: {len(target_cloud)} points from {len(self.cloud_window)} frames")

        rospy.loginfo("Starting ICP registration...")
        transformation, final_error = icp_2d(
            self.source_cloud_2d,
            target_cloud,
            max_iterations=self.max_iterations,
            tolerance=1e-6,
            max_distance=self.max_distance,
            init_x=self.init_x,
            init_y=self.init_y,
            init_yaw=self.init_yaw,
            verbose=True
        )
        # transformation1 = np.copy(transformation)
        # transformation1[:3, :3] = np.eye(3)
        # transformation1[0, 3] = 14
        # transformation1[1, 3] = 4
        # 对source_cloud_3d进行变换并发布
        source_transformed = transform_3d_points(self.source_cloud_3d, transformation)
        cloud_msg = create_point_cloud_msg(source_transformed, frame_id="camera_init")
        self.transformed_cloud_pub.publish(cloud_msg)
        rospy.loginfo(f"Published transformed source cloud: {len(source_transformed)} points, error: {final_error:.6f}")


if __name__ == "__main__":
    node = ICP2DRegistrationNode()
    rospy.spin()