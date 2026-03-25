#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Imu
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import open3d as o3d

class initPosePubNode:
    def __init__(self):
        rospy.init_node('initPosePub', anonymous=True)
        # --- Parameters ---
        self.accumulate_num = rospy.get_param('~accumulate_num', 5)
        # --- Topic names ---
        self.topic_pcd = rospy.get_param('~topic_pc', 'cloud_registered_body')
        self.topic_pose = rospy.get_param('~topic_init_pose', 'init_pose')
        self.topic_imu = rospy.get_param('common/imu_topic', '/quad_0/imu')        
        self.T_init = np.eye(4)
        self.T_init[0,3] = rospy.get_param('~init_x', '0.0')
        self.T_init[1,3] = rospy.get_param('~init_y', '0.0')
        init_yaw = rospy.get_param('~init_yaw', '0.0')        
        self.T_init[:3, :3] = R.from_euler('z', init_yaw).as_matrix()

        # --- Buffer ---
        self.cloud_buffer = []  # Store point cloud data
        self.buffer_lock = False  # Processing lock
        self.imu_buffer = []  # Store IMU data: [(timestamp, linear_acceleration), ...]
        pcd_path = rospy.get_param('mapping/map_path', '')
        pcd = o3d.io.read_point_cloud(pcd_path)
        self.points_db = np.asarray(pcd.points)    

        # --- ROS Interface ---
        self.sub_pcd = rospy.Subscriber(self.topic_pcd, PointCloud2, self.pcd_callback, queue_size=10)
        self.sub_imu = rospy.Subscriber(self.topic_imu, Imu, self.imu_callback, queue_size=100)
        self.pub_pose = rospy.Publisher(self.topic_pose, PoseStamped, queue_size=10)


    def imu_callback(self, msg):
        """IMU callback: store linear acceleration and timestamp"""
        if self.buffer_lock: return            
        timestamp = msg.header.stamp.to_sec()
        linear_acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        self.imu_buffer.append((timestamp, linear_acc))

    def estimate_attitude_from_imu(self):
        """
        Estimate rotation matrix from IMU accelerometer data.
        Assumption: IMU is static, so measured acceleration = gravity.

        Method: Use cross product to get rotation axis, then Rodrigues formula.
        - v = normalize(acc_mean) is gravity direction in body frame
        - target = [0, 0, -1] is gravity direction in world frame
        - R @ target = v

        Note: Yaw is unobservable from accelerometer alone.

        Returns:
            R: 3x3 rotation matrix, or None if buffer is empty
        """
        if len(self.imu_buffer) == 0:
            rospy.logwarn("IMU buffer is empty, cannot estimate attitude")
            return None

        # 1. Extract accelerations and compute mean
        acc_data = np.array([item[1] for item in self.imu_buffer])  # (N, 3)
        acc_mean = np.mean(acc_data, axis=0)  # (3,)

        # 2. Normalize gravity vectors
        v = acc_mean / np.linalg.norm(acc_mean)  # gravity in body frame
        target = np.array([0.0, 0.0, 1.0])     # gravity in world frame

        # 3. Compute rotation via cross product
        cross = np.cross(v, target)
        cos_angle = np.dot(v, target)
        # Handle near-parallel case
        if np.linalg.norm(cross) < 1e-6:
            if cos_angle > 0: R_imu = np.eye(3) # Vectors are parallel, no rotation needed                
            else: R_imu = np.diag([1.0, -1.0, -1.0]) # Vectors are opposite, rotate 180° around x-axis   
        else:
            axis = cross / np.linalg.norm(cross)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            R_imu = R.from_rotvec(axis * angle).as_matrix()
        euler = R.from_matrix(R_imu).as_euler('zyx', degrees=True)
        rospy.loginfo(f"Estimated attitude: roll={euler[2]:.2f}deg, pitch={euler[1]:.2f}deg, yaw={euler[0]:.2f}deg")
        return R_imu

    def icp_refine(self, query_pts, T_init, db_pts, max_corr_dist=1.0, max_iter=50):
        """
        ICP refinement for point cloud registration.

        Args:
            query_pts: (N, 3) query point cloud
            T_init: (4, 4) initial transformation matrix
            db_pts: (M, 3) database point cloud
            max_corr_dist: max correspondence distance for ICP
            max_iter: max iterations for ICP

        Returns:
            T_icp: (4, 4) refined transformation matrix, or T_init if ICP fails
        """
        # Transform query points with initial estimate
        query_pts_h = np.hstack([query_pts, np.ones((query_pts.shape[0], 1))])
        query_pts_init = (T_init @ query_pts_h.T).T[:, :3]

        # Create Open3D point clouds
        query_pcd = o3d.geometry.PointCloud()
        query_pcd.points = o3d.utility.Vector3dVector(query_pts_init)

        db_pcd = o3d.geometry.PointCloud()
        db_pcd.points = o3d.utility.Vector3dVector(db_pts)

        # ICP registration
        icp_result = o3d.pipelines.registration.registration_icp(
            query_pcd, db_pcd,
            max_correspondence_distance=max_corr_dist,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )

        rospy.loginfo(f"ICP result: fitness={icp_result.fitness:.4f}, rmse={icp_result.inlier_rmse:.4f}")

        return icp_result.transformation

    def pcd_callback(self, msg):
        """Point cloud callback: accumulate frames and estimate pose against database"""
        if self.buffer_lock: return            

        # 1. Parse point cloud
        pcd_iter = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        pcd_numpy = np.array(list(pcd_iter), dtype=np.float32)

        if pcd_numpy.shape[0] == 0:
            rospy.logwarn("Empty point cloud received, skipping...")
            return

        # 2. Add to buffer
        self.cloud_buffer.append(pcd_numpy)
        rospy.loginfo(f"Received frame {len(self.cloud_buffer)}/{self.accumulate_num}, points: {pcd_numpy.shape[0]}")

        # 3. Check if accumulated enough frames
        if len(self.cloud_buffer) >= self.accumulate_num and len(self.imu_buffer)>100: 
            self.buffer_lock = True
            # start_time = time.time()

            R_imu = self.estimate_attitude_from_imu()  # Optional: use IMU to estimate initial attitude (not used in current pipeline)

            # 4. Merge point clouds
            accumulated_cloud = np.vstack(self.cloud_buffer)
            rospy.loginfo(f"Accumulated {len(self.cloud_buffer)} frames, total points: {accumulated_cloud.shape[0]}")

            # 4.5 Project to horizontal plane using IMU attitude
            pts = accumulated_cloud[:, :3]  # xyz
            intensity = accumulated_cloud[:, 3:4]  # intensity
            pts_horizon = (R_imu @ pts.T).T  # Rotate to horizontal frame
            accumulated_cloud = np.hstack([pts_horizon, intensity])
            T_icp = self.icp_refine(pts_horizon, self.T_init, self.points_db, max_corr_dist=1.0, max_iter=50)
            T_imu = np.eye(4)
            T_imu[:3, :3] = R_imu
            T_est = T_icp @ self.T_init @ T_imu
            self.publish_relative_pose(msg.header, T_est)
            # 9. Clear buffer
            self.cloud_buffer.clear()
            # self.buffer_lock = False

    def publish_relative_pose(self, header, H):
        """Publish relative pose"""
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = "relative_pose"

        R_= R.from_matrix(H[:3, :3])
        quat = R_.as_quat()
        euler_angles = R_.as_euler('xyz', degrees=True)

        pose_msg.pose.position.x = H[0, 3]
        pose_msg.pose.position.y = H[1, 3]
        pose_msg.pose.position.z = H[2, 3]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pub_pose.publish(pose_msg)
        rospy.loginfo(f"Published relative pose: tx={H[0, 3]:.2f}m, ty={H[1, 3]:.2f}m, tz={H[2, 3]:.2f}m, \
                    roll={euler_angles[0]:.2f}deg , pitch={euler_angles[1]:.2f}deg, yaw={euler_angles[2]:.2f}deg")


if __name__ == "__main__":
    node = initPosePubNode()
    rospy.spin()