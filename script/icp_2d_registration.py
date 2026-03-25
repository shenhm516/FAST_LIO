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


def voxel_downsample_2d(
    points: np.ndarray,
    voxel_size: float
) -> np.ndarray:
    """
    对2D点云进行体素下采样

    Args:
        points: 2D点云数组 (N, 2)
        voxel_size: 体素大小

    Returns:
        下采样后的点云数组 (M, 2)
    """
    if len(points) == 0:
        return points

    # 计算每个点所属的体素索引
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # 使用字典存储每个体素中的一个点
    voxel_dict = {}
    for i, idx in enumerate(voxel_indices):
        key = (idx[0], idx[1])
        if key not in voxel_dict:
            voxel_dict[key] = points[i]

    return np.array(list(voxel_dict.values()))


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
        self.voxel_size = rospy.get_param('~voxel_size', 0.1)

        # 滑窗参数
        self.window_size = rospy.get_param('~window_size', 10)
        self.cloud_window = deque(maxlen=self.window_size)  # 存储最近的N帧2D点云

        # 初值参数
        self.init_x = rospy.get_param('~init_x', [])
        self.init_y = rospy.get_param('~init_y', [])
        self.init_yaw = rospy.get_param('~init_yaw', [])        

        # 读取PCD文件作为源点云
        self.source_cloud_2d = None
        self.source_cloud_3d = None
        pcd_path = rospy.get_param('~pcd_path', [])
        self.finish_flag = [False] * len(pcd_path)

        self.source_cloud_3d = []
        self.source_cloud_2d = []
        for i in range(len(pcd_path)):
            pcd = o3d.io.read_point_cloud(pcd_path[i])
            points_3d = np.asarray(pcd.points)
            self.source_cloud_3d.append(points_3d)
            self.source_cloud_2d.append(extract_2d_slice(
                points_3d,
                z_center=self.z_center,
                z_tolerance=self.z_tolerance,
                max_points=self.max_points
            ))
        #     print(self.source_cloud_2d[i].shape)
        # exit()

        # 订阅点云话题
        self.subscriber = rospy.Subscriber('cloud_registered', PointCloud2, self.cloud_callback, queue_size=1)

        # 发布变换后的点云
        self.transformed_cloud_pub = rospy.Publisher('cloud_house', PointCloud2, queue_size=1)

        # rospy.loginfo(f"ICP 2D Registration Node initialized")
        # rospy.loginfo(f"z_center: {self.z_center}m, z_tolerance: {self.z_tolerance}m")
        # rospy.loginfo(f"Sliding window size: {self.window_size}")
        # rospy.loginfo(f"Initial pose: x={self.init_x}, y={self.init_y}, yaw={np.degrees(self.init_yaw):.2f}°")

    def cloud_callback(self, msg: PointCloud2):
        """
        点云回调函数

        Args:
            msg: PointCloud2消息
        """
        if all(self.finish_flag): return

        # 将PointCloud2转换为numpy数组
        points_3d = []
        for point in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            points_3d.append([point[0], point[1], point[2]])

        if len(points_3d) < 100:
            rospy.logwarn("Too few points received, skipping...")
            return

        points_3d = np.array(points_3d)
        # rospy.loginfo(f"Converted to numpy array: {points_3d.shape}")

        # 提取2D点云
        points_2d = extract_2d_slice(
            points_3d,
            z_center=self.z_center,
            z_tolerance=self.z_tolerance,
            max_points=self.max_points
        )
        # rospy.loginfo(f"Extracted 2D slice: {len(points_2d)} points (z: {self.z_center}±{self.z_tolerance}m)")

        if len(points_2d) < 10:
            rospy.logwarn("Too few points in 2D slice, skipping...")
            return

        # 将当前帧加入滑窗
        self.cloud_window.append(points_2d)
        # rospy.loginfo(f"Added to sliding window, current size: {len(self.cloud_window)}/{self.window_size}")

        # 如果滑窗未满，等待更多帧
        if len(self.cloud_window) < self.window_size:
            rospy.loginfo(f"Waiting for more frames ({len(self.cloud_window)}/{self.window_size})")
            return

        # 合并滑窗中的所有点云作为目标点云
        target_cloud = np.vstack(list(self.cloud_window))
        # rospy.loginfo(f"Combined target cloud: {len(target_cloud)} points from {len(self.cloud_window)} frames")

        rospy.loginfo("Starting ICP registration...")
        for house_id in range(len(self.source_cloud_2d)):
            if self.finish_flag[house_id] == True: continue

            # 体素下采样
            source_down = voxel_downsample_2d(self.source_cloud_2d[house_id], self.voxel_size)
            target_down = voxel_downsample_2d(target_cloud, self.voxel_size)

            for iter in range(3):
                max_distance = max(self.max_distance / (iter + 1), 0.1)
                # for house_id in range(len(self.source_cloud_2d)):
                if iter == 0:
                    init_x = self.init_x[house_id]
                    init_y = self.init_y[house_id]
                    init_yaw = self.init_yaw[house_id]
                transformation, score = self.icp_2d(
                    source_down,
                    target_down,
                    max_iterations=self.max_iterations,
                    tolerance=1e-6,
                    max_distance=max_distance,
                    init_x=init_x,
                    init_y=init_y,
                    init_yaw=init_yaw,
                    verbose=False
                )
                init_x = transformation[0,3]
                init_y = transformation[1,3]
                init_yaw, _, _ = Rotation.from_matrix(transformation[:3, :3]).as_euler('zyx')
                if max_distance == 0.1: break

            
            
            transformation_init = np.eye(4)
            transformation_init[:3, :3] = Rotation.from_euler('z', self.init_yaw[house_id]).as_matrix()
            transformation_init[0, 3] = self.init_x[house_id]
            transformation_init[1, 3] = self.init_y[house_id]
            transformation_error = np.linalg.inv(transformation)@transformation_init
            angle_error = np.linalg.norm(Rotation.from_matrix(transformation_error[:3,:3]).as_rotvec())
            t_error = np.linalg.norm(transformation_error[:2,3])
            print(house_id, score['score'], score['inlier_rmse'], score['inlier_count'], t_error, angle_error)
            if (score['score']>0.3 or score['inlier_count']>500) and score['inlier_rmse'] < 0.2 and t_error < 1.5 and angle_error<np.deg2rad(15.0):
                # transformation = np.eye(4)
                # transformation[:3, :3] = Rotation.from_euler('z', self.init_yaw[self.house_id]).as_matrix()
                # transformation[0, 3] = self.init_x[self.house_id]
                # transformation[1, 3] = self.init_y[self.house_id]
                source_transformed = transform_3d_points(self.source_cloud_3d[house_id], transformation)
                cloud_msg = create_point_cloud_msg(source_transformed, frame_id=str(house_id)) #str(house_id) camera_init
                self.transformed_cloud_pub.publish(cloud_msg)
                self.finish_flag[house_id] = True
                # self.house_id += 1
                # if self.house_id >= len(self.source_cloud_2d): self.finish_flag = True
                # rospy.loginfo(f"Published transformed source cloud: {len(source_transformed)} points, Score: {score['score']}")

    def icp_2d(self,
        source: np.ndarray,
        target: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        max_distance: float = 0.5,
        init_x: float = 0.0,
        init_y: float = 0.0,
        init_yaw: float = 0.0,
        verbose: bool = True
    ) -> Tuple[np.ndarray, dict]:
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

        # 计算评分指标 (参考Open3D)
        distances, _ = target_tree.query(current_source)
        inlier_mask = distances < self.max_distance
        inlier_count = int(np.sum(inlier_mask))

        # fitness: 内点占比
        fitness = inlier_count / len(source)
        # inlier_rmse: 仅计算内点的RMSE
        inlier_rmse = np.sqrt(np.mean(distances[inlier_mask] ** 2)) if inlier_count > 0 else float('inf')

        score = {
            'fitness': fitness,
            'inlier_rmse': inlier_rmse,
            'inlier_count': inlier_count,
            'score': fitness  # 综合评分使用fitness
        }

        if verbose:
            yaw_final = np.arctan2(R_total[1, 0], R_total[0, 0])
            rospy.loginfo(f"ICP completed: fitness={fitness:.4f}, inlier_rmse={inlier_rmse:.6f}")
            rospy.loginfo(f"  Yaw: {np.degrees(yaw_final):.2f}°")
            rospy.loginfo(f"  Translation: [{t_total[0]:.4f}, {t_total[1]:.4f}]")

        return transformation, score
    
if __name__ == "__main__":
    node = ICP2DRegistrationNode()
    rospy.spin()