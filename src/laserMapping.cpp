// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.hpp"
#include "ikd-Tree/ikd_Tree.h"
#include <unordered_map>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <algorithm>
#include <execution>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)
// #define USE_voxel

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
bool   load_offline_map = false;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, offline_map_path, imu_topic, imu_odom_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double win_beg_time = -1e6, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0;//, fov_deg = 0;
double cube_len = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   use_kernal = false;
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI>             lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
nav_msgs::Odometry odomImu;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

/*** Fast prediction ***/
double latest_time;
V3D latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
Eigen::Quaterniond latest_Q;
ros::Publisher pubOdomImu;

/*** Voxel map ***/
double rootSurfVoxelSize;
vector<unordered_map<VOXEL_LOC, OCTO_TREE*>::iterator> surfhashKeyMargVector;  
vector<unordered_map<VOXEL_LOC, OCTO_TREE*>::iterator> feat_map_update_iter;
unordered_map<VOXEL_LOC, OCTO_TREE*> surf_map;

/*** Segment point cloud ***/
PointCloudXYZI::Ptr  ptr_seg(new PointCloudXYZI());
double lidar_mean_scantime = 0.1;
ros::Publisher pubLaserCloudSeg;

V3D last_P_cur(Zero3d);
V3D last_V_cur(Zero3d);
Eigen::Quaternion<double> last_Q_cur;

std::vector<std::deque<LidarMsgGroup>> lidar_msg_buffer;
int fix_rate = 50, frame_num = 0;
ofstream fout_pre, fout_out, fout_dbg;
ros::Publisher pubOdomAftMapped, pubPath, pubLaserCloudFull, pubLaserCloudFull_body;
double aver_time_incre = 0, aver_time_const_H_time = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_consu = 0, aver_time_solve = 0;
FILE *fp;

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle 1-3
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos 4-6 
    fprintf(fp, "%lf %lf %lf ", state_point.omg(0), state_point.omg(1), state_point.omg(2)); // omega 7-9 
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel 10-12 
    fprintf(fp, "%lf %lf %lf ", state_point.acc(0), state_point.acc(1), state_point.acc(2)); // Acc 13-15 
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g 16-18 
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a 19-21 
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // grav 22-24  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    double dt = pi->curvature/double(1000);
    V3D p_global(state_point.rot.toRotationMatrix() * Exp(state_point.omg, dt)* (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos + state_point.vel*dt + 0.5*state_point.acc*dt*dt);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


// template<typename T>
// void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
// {
//     V3D p_body(pi[0], pi[1], pi[2]);
//     V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

//     po[0] = p_global(0);
//     po[1] = p_global(1);
//     po[2] = p_global(2);
// }

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;   
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void fastPredictIMU(double t, V3D linear_acceleration, V3D angular_velocity)
{
    double dt = t - latest_time;
    // std::cout << dt << std::endl;
    latest_time = t;
    V3D un_gyr = 0.5 * (latest_gyr_0 + angular_velocity - latest_Bg);
    Eigen::Quaternion<double> dq(Exp(un_gyr, dt));
    latest_Q = dq * latest_Q;
    V3D un_acc_1 = latest_Q * (linear_acceleration - latest_Ba);
    for (int i = 0; i < 3; i++) un_acc_1[i] += state_point.grav[i]; //has some problem
    V3D un_acc = 0.5 * (latest_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = un_acc_1;
    latest_gyr_0 = angular_velocity - latest_Bg;
}

void publish_odometry_imu(double time_stamp, const ros::Publisher & pubOdomImu)
{
    nav_msgs::Odometry odometry;
    odometry.header.frame_id = "camera_init";
    odometry.child_frame_id = "body";
    odometry.header.stamp = ros::Time().fromSec(time_stamp);
    
    odometry.pose.pose.position.x = latest_P.x();
    odometry.pose.pose.position.y = latest_P.y();
    odometry.pose.pose.position.z = latest_P.z();
    odometry.pose.pose.orientation.x = latest_Q.x();
    odometry.pose.pose.orientation.y = latest_Q.y();
    odometry.pose.pose.orientation.z = latest_Q.z();
    odometry.pose.pose.orientation.w = latest_Q.w();
    odometry.twist.twist.linear.x = latest_V.x();
    odometry.twist.twist.linear.y = latest_V.y();
    odometry.twist.twist.linear.z = latest_V.z();
    pubOdomImu.publish(odometry);
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg, const int& lidar_id) {
    mtx_buffer.lock();
    // printf("%.6f %d\n", msg->header.stamp.toSec(), lidar_id);
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr, lidar_id);
    sort(ptr->points.begin(), ptr->points.end(), time_list);    
    LidarMsgGroup lidar_msg;
    lidar_msg.msg_beg_time = msg->header.stamp.toSec();
    if (ptr->points.size() > 0) {
        lidar_msg.msg_end_time = lidar_msg.msg_beg_time + ptr->points[ptr->points.size()-1].curvature / double(1000);
        lidar_msg.point_beg_time = lidar_msg.msg_beg_time;
    } else return;
    lidar_msg.cloud = *ptr;
    lidar_msg.lidar_id = lidar_id;
    lidar_msg_buffer[lidar_id].push_back(lidar_msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
// {
//     mtx_buffer.lock();
//     scan_count ++;
//     double preprocess_start_time = omp_get_wtime();
//     // if (msg->header.stamp.toSec() < last_timestamp_lidar)
//     // {
//     //     ROS_ERROR("lidar loop back, clear buffer");
//     //     lidar_buffer.clear();
//     // }

//     PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
//     p_pre->process(msg, ptr);
//     sort(ptr->points.begin(), ptr->points.end(), time_list);
//     static double last_timestamp_lidar_ = msg->header.stamp.toSec();
//     // time_buffer.push_back(last_timestamp_lidar);
//     // double cnt = 0;
//     for (int i = 0; i < ptr->size(); i++) {
//         auto pt = ptr->points[i];
//         auto time_pt = msg->header.stamp.toSec() + pt.curvature / double(1000);
//         // printf("%.6f \n", time_pt);
//         // std::cout << pt.curvature << std::endl;
//         if (time_pt > last_timestamp_lidar_ && time_pt <= last_timestamp_lidar_ + lidar_mean_scantime) {
//             pt.curvature = 1000.0*(time_pt-last_timestamp_lidar_); //ms
//             // if (ptr_seg->size()==0) printf("%.6f \n", time_pt);
//             ptr_seg->push_back(pt);
//             // cnt++;
//             // std::cout << ptr_seg->size() << std::endl;
//         } else if (time_pt > last_timestamp_lidar_ + lidar_mean_scantime) {             
//             PointCloudXYZI::Ptr ptr_div_i(new PointCloudXYZI());
//             *ptr_div_i = *ptr_seg;
//             // printf("%.6f \n", ptr_div_i->points[0].curvature);
//             // std::cout << ptr_div_i->size() << " " << ptr_seg->size() << std::endl;
//             lidar_buffer.push_back(*ptr_div_i);            
//             ptr_seg->clear();
//             time_buffer.push_back(last_timestamp_lidar_);
//             // printf("%.6f \n", ptr_seg->points[i]);
//             // sensor_msgs::PointCloud2 laserCloudSeg;
//             // pcl::toROSMsg(*ptr_div_i, laserCloudSeg);
//             // laserCloudSeg.header.stamp = ros::Time().fromSec(last_timestamp_lidar);
//             // laserCloudSeg.header.frame_id = "body";
//             // pubLaserCloudSeg.publish(laserCloudSeg);
            
//             // std::cout << cnt << std::endl;
//             // cnt = 0;
//             last_timestamp_lidar_ += lidar_mean_scantime;            
//             // cnt++;
//         } 
//     }
//     // printf("%.6f \n", last_timestamp_lidar);
//     // std::cout << last_timestamp_lidar << std::endl;
//     // std::cout << cnt << " " << ptr->size() << std::endl;
//     // std::cout << cnt << " " << lidar_mean_scantime << std::endl;
//     // lidar_buffer.push_back(ptr);
//     // time_buffer.push_back(msg->header.stamp.toSec());
//     // last_timestamp_lidar = msg->header.stamp.toSec();
//     s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
//     mtx_buffer.unlock();
//     sig_buffer.notify_all();
// }

void psa_rs_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg, const int& lidar_id) {
    mtx_buffer.lock();
    // printf("%.6f %d\n", msg->header.stamp.toSec(), lidar_id);
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr, lidar_id);
    sort(ptr->points.begin(), ptr->points.end(), time_list);    
    LidarMsgGroup lidar_msg;
    // lidar_msg.msg_end_time = msg->header.stamp.toSec();
    if (ptr->points.size()) {
        // std::cout << ptr->points[0].curvature*1e-3 << " " << ptr->points[ptr->points.size()-1].curvature*1e-3<< std::endl;
        lidar_msg.msg_end_time = msg->header.stamp.toSec() + ptr->points[ptr->points.size()-1].curvature*1e-3;
        lidar_msg.msg_beg_time = msg->header.stamp.toSec() + ptr->points[0].curvature*1e-3;
        lidar_msg.point_beg_time = lidar_msg.msg_beg_time;
    } else {
        mtx_buffer.unlock();
        return;
    }
    // printf("%.6f %.6f\n", lidar_msg.msg_beg_time, lidar_msg.msg_end_time);
    lidar_msg.cloud = *ptr;
    lidar_msg.lidar_id = lidar_id;
    lidar_msg_buffer[lidar_id].push_back(lidar_msg);

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
// bool   timediff_set_flg = false;
// void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
// {
//     mtx_buffer.lock();
//     double preprocess_start_time = omp_get_wtime();
//     scan_count ++;
//     // if (msg->header.stamp.toSec() < last_timestamp_lidar)
//     // {
//     //     ROS_ERROR("lidar loop back, clear buffer");
//     //     lidar_buffer.clear();
//     // }
//     // last_timestamp_lidar = msg->header.stamp.toSec();
//     static double last_timestamp_lidar_ = msg->header.stamp.toSec();
    
//     if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar_) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
//     {
//         printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar_);
//     }

//     if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar_ - last_timestamp_imu) > 1 && !imu_buffer.empty())
//     {
//         timediff_set_flg = true;
//         timediff_lidar_wrt_imu = last_timestamp_lidar_ + 0.1 - last_timestamp_imu;
//         printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
//     }

//     PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
//     p_pre->process(msg, ptr);
//     lidar_buffer.push_back(*ptr);
//     time_buffer.push_back(last_timestamp_lidar_);
    
//     s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
//     mtx_buffer.unlock();
//     sig_buffer.notify_all();
// }

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    // sensor_msgs::Imu::Ptr msg_inv(new sensor_msgs::Imu(*msg));
    // msg_inv->linear_acceleration.y *= -1.0;
    // msg_inv->linear_acceleration.z *= -1.0;
    // msg_inv->angular_velocity.y *= -1.0;
    // msg_inv->angular_velocity.z *= -1.0;

    imu_buffer.push_back(msg);
    // imu_buffer.push_back(msg_inv);
    // V3D linearAcceleration(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    // V3D angularVelocity(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    // fastPredictIMU(timestamp, linearAcceleration, angularVelocity);
    // publish_odometry_imu(timestamp, pubOdomImu);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// double lidar_mean_scantime = 0.0;
// int    scan_num = 0;
// bool sync_packages_(MeasureGroup &meas) {
//     // std::cout << "Hello 0" << std::endl;
//     int pc_size = 0;
//     for (int i=0; i<lidar_num; i++) {
//         pc_size += lidar_msg_buffer[i].size(); //num of lidar msg in buffer for i-th lidar        
//     }
//     if (pc_size==0 || imu_buffer.empty()) {
//         // std::cout << "Hello 0.1" << std::endl;
//         win_beg_time += lidar_mean_scantime;
//         return false;
//     }
//     // std::cout << "Hello 0.2" << std::endl;
//     // if (flg_first_scan) {
//     for (int i=0; i<lidar_num; i++) {
//         if (lidar_msg_buffer[i].size()) {
//             LidarMsgGroup lidar_msg = lidar_msg_buffer[i].front();
//             if (lidar_msg.msg_beg_time < win_beg_time || win_beg_time<0.0)
//                 win_beg_time = lidar_msg.msg_beg_time; //use the minimal time stamp as the start time
//         }
//     }
//     //     first_lidar_time = win_beg_time;
//     //     p_imu->first_lidar_time = first_lidar_time;
//     //     flg_first_scan = false;
//     //     // return false;
//     // }
//     // std::cout << "Hello 0.3" << std::endl;
//     /*** push a lidar scan ***/
//     meas.lidar_beg_time = win_beg_time;
//     lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
//     meas.lidar_end_time = lidar_end_time;

//     // if (last_timestamp_imu < lidar_end_time)
//     // {
//     //     printf("bye %.6f %.6f \n", last_timestamp_imu, lidar_end_time);
//     //     // std::cout << "Hello !!!!!!!!!!!!!!!!!" << last_timestamp_imu << " " << lidar_end_time << std::endl;
//     //     win_beg_time += lidar_mean_scantime;
//     //     return false;
//     // }

//     for (int i = 0; i < lidar_msg_buffer[0].front().cloud.size(); i++) {
//         auto pt = lidar_msg_buffer[0].front().cloud.points[i];
//         auto time_pt = lidar_msg_buffer[0].front().msg_beg_time + pt.curvature / double(1000);
//         // // printf("%.6f \n", time_pt);
//         // // std::cout << pt.curvature << std::endl;
//         if (time_pt > win_beg_time && time_pt <= lidar_end_time) {
//             pt.curvature = 1000.0*(time_pt-lidar_msg_buffer[0].front().msg_beg_time); //ms
//             // if (ptr_seg->size()==0) printf("%.6f \n", time_pt);
//             ptr_seg->push_back(pt);
//             // cnt++;
//             // std::cout << ptr_seg->size() << std::endl;
//         } else if (time_pt > lidar_end_time) {
//             PointCloudXYZI::Ptr ptr_div_i(new PointCloudXYZI());
//             *ptr_div_i = *ptr_seg;
//             meas.lidar = *ptr_div_i;
//             // printf("%.6f \n", ptr_div_i->points[0].curvature);
//             // std::cout << ptr_div_i->size() << " " << ptr_seg->size() << std::endl;
//             // lidar_buffer.push_back(*ptr_div_i);            
//             ptr_seg->clear();
//             break;
//             // printf("%.6f \n", ptr_seg->points[i]);
//             // sensor_msgs::PointCloud2 laserCloudSeg;
//             // pcl::toROSMsg(*ptr_div_i, laserCloudSeg);
//             // laserCloudSeg.header.stamp = ros::Time().fromSec(last_timestamp_lidar);
//             // laserCloudSeg.header.frame_id = "body";
//             // pubLaserCloudSeg.publish(laserCloudSeg);
            
//             // std::cout << cnt << std::endl;
//             // cnt = 0;
//             // last_timestamp_lidar_ += lidar_mean_scantime;            
//             // cnt++;
//         } 
//     }

    
//     // std::cout << "Hello 0.4" << std::endl;

//     double imu_time = imu_buffer.front()->header.stamp.toSec();
//     meas.imu.clear();
//     meas.imu.swap(meas.imu_cur);
//     meas.imu_cur.clear();
    
//     // std::cout << "Hello 0.5" << std::endl;
//     printf("Hello %.6f %.6f %.6f %.6f %d\n", imu_time, imu_buffer.back()->header.stamp.toSec(), meas.lidar_beg_time, meas.lidar_end_time, meas.lidar.size());
   
//     while ((!imu_buffer.empty()) && (imu_time < meas.lidar_end_time))
//     {
//         // std::cout << "!!!!!!!!!" << std::endl;
//         if(imu_time > meas.lidar_beg_time) meas.imu_cur.push_back(imu_buffer.front());
//         else meas.imu.push_back(imu_buffer.front()); //shm: only effected in the first time
//         imu_buffer.pop_front();
//         // std::cout << "Hello 0.51" << std::endl;
//         if (imu_buffer.empty()) break;
//         else imu_time = imu_buffer.front()->header.stamp.toSec();
//         // std::cout << "Hello 0.52" << std::endl;
//     }
//     std::cout << meas.imu.size() << " " << meas.imu_cur.size() << std::endl;
//     if (lidar_msg_buffer[0].front().msg_end_time<lidar_end_time) lidar_msg_buffer[0].pop_front();
//     win_beg_time += lidar_mean_scantime;
//     // std::cout << "Hello 1" << std::endl;
//     return true;
// }

bool sync_packages(MeasureGroup &meas)
{    
    // std::cout << "Hello 1" << std::endl;    
    if (win_beg_time<0.0) {
        for (int i=0; i<lidar_num; i++) {
            if (lidar_msg_buffer[i].size()) {
                LidarMsgGroup lidar_msg = lidar_msg_buffer[i].front();
                if (lidar_msg.point_beg_time < win_beg_time || win_beg_time<0.0)
                    win_beg_time = lidar_msg.point_beg_time; //use the minimal time stamp as the start time
            }
        }
    }
    // std::vector<double> lidar_time;
    // for (int i=0; i<lidar_num; i++) {
    //     if (lidar_msg_buffer[i].size()) {
    //         lidar_time.push_back(lidar_msg_buffer[i].back().point_beg_time);
    //     }
    // }
    // if (lidar_time.size()) {
    //     auto minTime = std::min_element(lidar_time.begin(), lidar_time.end());
    //     win_beg_time = *minTime;
    // }
    // else {
    //     // std::cout << "Hello 1.5" << std::endl;    
    //     return false;
    // }
    

    // if (win_beg_time<0) win_beg_time = lidar_end_time;

    // std::cout << "Hello -1" << std::endl;
    lidar_end_time = win_beg_time + lidar_mean_scantime;
    int pc_size = 0;
    for (int i=0; i<lidar_num; i++) {
        // // int msg_index = 0;
        while(!lidar_msg_buffer[i].empty()) {
            // std::cout << "Hello 1.5" << std::endl;
            // if (lidar_msg_buffer[i].size()==0) break;
            if (lidar_msg_buffer[i].front().msg_end_time < win_beg_time) {
                lidar_msg_buffer[i].pop_front();
            } else {
                // printf("%.6f %.6f %.6f\n",  lidar_msg_buffer[i].front().point_beg_time, 
                //                             lidar_msg_buffer[i].front().msg_end_time, 
                //                             win_beg_time);
                if (lidar_msg_buffer[i].front().msg_end_time > win_beg_time && 
                    lidar_msg_buffer[i].front().point_beg_time <= lidar_end_time) 
                    pc_size++;
                break;
            }
        }
        // pc_size += lidar_msg_buffer[i].size(); //num of lidar msg in buffer for i-th lidar        
    }
    if (pc_size==0 || imu_buffer.empty()) {
        // printf("%.6f %.6f %.6f %.6f %d\n", lidar_msg_buffer[0].front().point_beg_time, 
        //                                 lidar_msg_buffer[0].front().msg_end_time, 
        //                                 win_beg_time, lidar_end_time, lidar_msg_buffer[0].size());
        // printf("%d %d\n", pc_size, imu_buffer.size());
        // std::cout << "Hello 0.1" << std::endl;

        // for (int i=0; i<lidar_num; i++) {
        //     if (lidar_msg_buffer[i].size()) {
        //         LidarMsgGroup lidar_msg = lidar_msg_buffer[i].front();
        //         if (lidar_msg.point_beg_time < win_beg_time || win_beg_time<0.0)
        //             win_beg_time = lidar_msg.point_beg_time; //use the minimal time stamp as the start time
        //     }
        // }
        int fail_num = 0;
        for (int lidar_id=0; lidar_id<lidar_num; lidar_id++) {
            if (lidar_msg_buffer[lidar_id].front().point_beg_time > lidar_end_time) {
                fail_num++;
            }
        }
        if(fail_num==lidar_num) {
            // std::cout << "Hello" << std::endl;
            win_beg_time = lidar_end_time;
        }
        // win_beg_time = lidar_end_time;
        // std::cout << "Hello 2" << std::endl;
        
        return false;
    }
    // if (lidar_buffer.empty() || imu_buffer.empty()) {
    //     // std::cout << "Hello 0 " << lidar_buffer.size() << " " << imu_buffer.size() << std::endl;
    //     return false;
    // }
    // win_beg_time = -1.0;


    

    // double pt_st = lidar_msg_buffer[0].front().msg_beg_time + lidar_msg_buffer[0].front().cloud.points[0].curvature / double(1000);
    // double pt_et = lidar_msg_buffer[0].front().msg_beg_time + lidar_msg_buffer[0].front().cloud.points[lidar_msg_buffer[0].front().cloud.size()-1].curvature / double(1000);
    // printf("%.6f %.6f %.6f %.6f\n", pt_st,pt_et, win_beg_time, lidar_end_time);
    /*** push a lidar scan ***/
    // std::cout << "Hello 0" << std::endl;
    // printf("Hello  %.6f %.6f\n", lidar_msg_buffer[0].front().point_beg_time, lidar_msg_buffer[0].front().msg_end_time);
    for (int lidar_id=0; lidar_id<lidar_num; lidar_id++) {
        meas.lidar[lidar_id].clear();
        bool loop_finish = true;        
        while(loop_finish && !lidar_msg_buffer[lidar_id].empty()) {
            // std::cout << "Hello 2.5 "  << loop_finish << " " << lidar_msg_buffer[lidar_id].size() << std::endl;
            // int numm = 0;
            // printf("1: %d\n", lidar_msg_buffer[lidar_id].front().cloud.size());
            for (int i = 0; i < lidar_msg_buffer[lidar_id].front().cloud.size(); i++) {                
                auto pt = lidar_msg_buffer[lidar_id].front().cloud.points[i];
                if (p_pre->lidar_type[lidar_id]==ROBOSENSE) {
                    // pt.curvature = 0.0;
                    pt.curvature = (pt.curvature*1e-3 + lidar_msg_buffer[lidar_id].front().msg_end_time - lidar_msg_buffer[lidar_id].front().msg_beg_time)*1e3;
                    // std::cout << pt.curvature << std::endl;
                } 
                auto time_pt = lidar_msg_buffer[lidar_id].front().msg_beg_time + pt.curvature / double(1000);
                
                // // std::cout << pt.curvature << std::endl;
                // printf("%.6f %.6f %.6f\n",time_pt,win_beg_time,lidar_end_time);
                if (time_pt >= win_beg_time && time_pt <= lidar_end_time) {
                    pt.curvature = 1000.0*(time_pt-win_beg_time); //ms
                    // printf("%f\n", pt.curvature);
                    // if (ptr_seg->size()==0)
                    //  printf("%.6f %.6f\n", time_pt, pt.curvature);
                    ptr_seg->push_back(pt);
                    // numm++;
                    if (i==lidar_msg_buffer[lidar_id].front().cloud.size()-1) {
                        meas.lidar[lidar_id] += *ptr_seg;    
                        ptr_seg->clear();
                        // std::cout << "Hello 0" << std::endl;
                        lidar_msg_buffer[lidar_id].pop_front();
                        break;
                        // std::cout << "Hello 1" << std::endl;
                    }
                } else if (time_pt > lidar_end_time) {
                    meas.lidar[lidar_id] += *ptr_seg;    
                    ptr_seg->clear();
                    if (i!=(lidar_msg_buffer[lidar_id].front().cloud.size()-1))
                    {
                        std::vector<int> index;
                        for(int j=i; j<lidar_msg_buffer[lidar_id].front().cloud.size(); j++) index.push_back(j);
                        // std::cout << "input: " << lidar_msg_buffer[0].front().cloud.size() << std::endl;
                        pcl::copyPointCloud(lidar_msg_buffer[lidar_id].front().cloud, index, lidar_msg_buffer[lidar_id].front().cloud);
                        // numm+=lidar_msg_buffer[lidar_id].front().cloud.size();
                        // std::cout << "output: " << lidar_msg_buffer[0].front().cloud.size() << std::endl;
                        lidar_msg_buffer[lidar_id].front().point_beg_time = lidar_msg_buffer[lidar_id].front().msg_beg_time + pt.curvature / double(1000);
                    } else {
                        lidar_msg_buffer[lidar_id].pop_front();
                    }
                    loop_finish = false;
                    break;
                } else if (i==lidar_msg_buffer[lidar_id].front().cloud.size()-1) {
                    lidar_msg_buffer[lidar_id].pop_front();
                    break;
                }
                
                // else numm++;
                // printf("%.6f %.6f %.6f %f\n",time_pt,win_beg_time,lidar_end_time, pt.curvature);
                
                // else {
                //     numm++;
                //     if (numm==lidar_msg_buffer[lidar_id].front().cloud.size()) {
                //         lidar_msg_buffer[lidar_id].pop_front();
                //     }
                // }
            }
            // if (numm>10) printf("2: %d\n", numm);

        }
    }   
    // std::cout << "Hello 1" << std::endl;
    // meas.lidar = lidar_buffer.front();
    // meas.lidar_beg_time = time_buffer.front();
    meas.lidar_beg_time = win_beg_time;
    // lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
    meas.lidar_end_time = lidar_end_time;
    win_beg_time = lidar_end_time;

    // if (last_timestamp_imu < lidar_end_time)
    // {
    //     printf("Hello %f.6 %f.6 \n", last_timestamp_imu, lidar_end_time);
    //     // std::cout << "Hello !!!!!!!!!!!!!!!!!" << last_timestamp_imu << " " << lidar_end_time << std::endl;
    //     return false;
    // }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    meas.imu.swap(meas.imu_cur);
    meas.imu_cur.clear();
    // printf("Hello %.6f %.6f %.6f %.6f %d\n", imu_time, imu_buffer.back()->header.stamp.toSec(), meas.lidar_beg_time, meas.lidar_end_time, meas.lidar.size());
    while ((!imu_buffer.empty()) && (imu_time < meas.lidar_end_time))
    {
        // std::cout << "Hello 3" << std::endl;
        if(imu_time > meas.lidar_beg_time) meas.imu_cur.push_back(imu_buffer.front());
        else meas.imu.push_back(imu_buffer.front()); //shm: only effected in the first time
        imu_buffer.pop_front();
        // std::cout << "Hello 1.1" << std::endl;
        if(!imu_buffer.empty()) imu_time = imu_buffer.front()->header.stamp.toSec();
        // std::cout << "Hello 1.2" << std::endl;
    }
    // std::cout << "Hello 2" << std::endl;
    // std::cout << meas.imu.size() << " input " << meas.imu_cur.size() << std::endl;

    // lidar_buffer.pop_front();
    // time_buffer.pop_front();
    // lidar_pushed = false;
    // std::cout << "Hello 3" << std::endl;
    
    return true;
}

// int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

#ifdef USE_voxel
void cut_voxel(std::unordered_map<VOXEL_LOC, OCTO_TREE*> &feat_map, pcl::PointCloud<PointType>::Ptr pl_feat, state_ikfom state) {
    // ros::WallTime starting_time = ros::WallTime::now();
    feat_map_update_iter.clear();
    std::for_each(std::execution::seq, pl_feat->points.begin(), pl_feat->points.end(), [](const auto& pt) {
        auto pt_w = pt;
        pointBodyToWorld(&(pt), &(pt_w));
        V3D pvec_tran(pt_w.x, pt_w.y, pt_w.z);
        float loc_xyz[3];
        for(int j=0; j<3; j++) {
            loc_xyz[j] = pvec_tran[j] / rootSurfVoxelSize;
            if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;        
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        // Find corresponding voxel
        auto iter = surf_map.find(position);//找到与当前特征对应的体素
        if(iter != surf_map.end()) {
            // iter->second->tmp_ori[frame_id].push_back(pvec_orig); // shm: add the original point into the silding window
            iter->second->Time = ros::Time::now().toSec();
            if (iter->second->octo_state==false && iter->second->plvec_tran->size()< 50) {
                // iter->second->Time = ros::Time::now().toSec();
                iter->second->plvec_tran->push_back(pvec_tran);                    
                if (iter->second->is2opt == false) {
                    feat_map_update_iter.push_back(iter);
                    iter->second->is2opt = true; //体素更新标志位
                }                    
            }
        }
        else {// If not finding, build a new voxel        
            OCTO_TREE *ot = new OCTO_TREE();    //建立一个新体素
            // ot->tmp_ori[frame_id].push_back(pvec_orig);  // shm: add the original point into the silding window
            ot->plvec_tran->push_back(pvec_tran);       //点云坐标(world fixed frame)
            // Voxel center coordinate
            ot->voxel_center[0] = (0.5+position.x) * rootSurfVoxelSize;
            ot->voxel_center[1] = (0.5+position.y) * rootSurfVoxelSize;
            ot->voxel_center[2] = (0.5+position.z) * rootSurfVoxelSize;
            ot->quater_length = rootSurfVoxelSize / 4.0; // A quater of side length
            // ot->correspondTime = ros::Time::now().toSec();
            // ot->is2opt = true;
            surf_map[position] = ot;
            feat_map_update_iter.push_back(surf_map.find(position));
            surfhashKeyMargVector.push_back(surf_map.find(position));
        }
    });

    // std::cout << ros::WallTime::now()- starting_time << std::endl;
    /****************根据新加入特征更新体素******************/
    std::for_each(std::execution::seq, feat_map_update_iter.begin(), feat_map_update_iter.end(), [](const auto& iter) {
        iter->second->root_centors.clear();
        iter->second->recut(0, 5, iter->second->root_centors, iter->second->pl_eigen, 5);
        iter->second->is2opt = false;
    });

    /*****************边缘化建立10s以上的体素（减少内存占用）******************/
    uint slowIndex = 0;
    for (uint i=0; i<surfhashKeyMargVector.size(); i++) {
        if (ros::Time::now().toSec()-surfhashKeyMargVector[i]->second->Time > 10){
            surfhashKeyMargVector[i]->second->octo_state = true;
            vector<Eigen::Vector3d>().swap(*surfhashKeyMargVector[i]->second->plvec_tran);
        } else {
            surfhashKeyMargVector[slowIndex++] = surfhashKeyMargVector[i];
        }
    }
    surfhashKeyMargVector.erase(surfhashKeyMargVector.begin()+slowIndex, surfhashKeyMargVector.end());
}
#endif

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.pose.position.x = state_point.pos_cur(0);
    out.pose.pose.position.y = state_point.pos_cur(1);
    out.pose.pose.position.z = state_point.pos_cur(2);
    out.twist.twist.linear.x = state_point.vel_cur(0);
    out.twist.twist.linear.y = state_point.vel_cur(1);
    out.twist.twist.linear.z = state_point.vel_cur(2);
    out.pose.pose.orientation.x = geoQuat.x;
    out.pose.pose.orientation.y = geoQuat.y;
    out.pose.pose.orientation.z = geoQuat.z;
    out.pose.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    // set_posestamp(msg_body_pose);
    msg_body_pose.pose.position.x = state_point.pos_cur(0);
    msg_body_pose.pose.position.y = state_point.pos_cur(1);
    msg_body_pose.pose.position.z = state_point.pos_cur(2);
    msg_body_pose.pose.orientation.x = geoQuat.x;
    msg_body_pose.pose.orientation.y = geoQuat.y;
    msg_body_pose.pose.orientation.z = geoQuat.z;
    msg_body_pose.pose.orientation.w = geoQuat.w;
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 
    // std::cout << "Hello H 0" << std::endl;
    /** closest surface search and residual computation **/
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];
        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        double dt = point_body.curvature/double(1000);
        V3D p_global(s.rot.toRotationMatrix() * Exp(s.omg,dt) * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos + s.vel*dt + 0.5*s.acc*dt*dt);
        
#ifdef USE_voxel
        float loc_xyz[3];            
        for(int j=0; j<3; j++) {
            loc_xyz[j] = p_global(j) / rootSurfVoxelSize;
            if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = surf_map.find(position);//找到与当前特征对应的体素
        point_selected_surf[i] = false;
        if(iter != surf_map.end() && iter->second->root_centors.size()) {
            float dist_record = 1e6;
            int index = -1;
            for (int j=0; j< iter->second->root_centors.size(); j++) {
                auto dist_x = iter->second->root_centors[j].x-p_global(0);
                auto dist_y = iter->second->root_centors[j].y-p_global(1);
                auto dist_z = iter->second->root_centors[j].z-p_global(2);
                auto dist = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);

                if (dist<dist_record) {
                    index = j;
                    dist_record = dist;
                }
            }
            if (index>=0) {
                PointType &ay = iter->second->root_centors[index];
                V3D center(ay.x, ay.y, ay.z);
                V3D direct(ay.normal_x, ay.normal_y, ay.normal_z);
                direct.normalize();
                double dista = fabs(direct.dot(p_global - center));
                if(dista <= 0.5) {
                    point_selected_surf[i] = true;
                    normvec->points[i].x = direct(0);
                    normvec->points[i].y = direct(1);
                    normvec->points[i].z = direct(2);
                    normvec->points[i].intensity = direct.dot(p_global - center);
                    res_last[i] = dista;
                }
            }
        }
#else        
        PointType &point_world = feats_down_world->points[i]; 
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;
        
        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;        
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
#endif
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    // if(!Measures.imu.empty()) {
    //     ekfom_data.h_x = MatrixXd::Zero(effct_feat_num+15, 33); //23 
    //     ekfom_data.h.resize(effct_feat_num+15);  //shm: this is the Z-h(x) vector, not h(x)
    // } else {
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num+18, 30); //23 
    ekfom_data.h.resize(effct_feat_num+18);  //shm: this is the Z-h(x) vector, not h(x)
    // }

    // std::cout << "Hello H 0.9" << std::endl; 
    // double max_dt = 0;
    // Eigen::Matrix<double, 3, 3> Hess_pos = Eigen::Matrix<double, 3, 3>::Zero();
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {
        // std::cout << "Hello H 0.2" << std::endl; 
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        double weight;
        if(use_kernal) weight = exp(-res_last[i]*res_last[i]/(2*0.01)); 
        else weight = 1.0;       
        ekfom_data.h_x.block<1, 3>(i,0) = weight*norm_vec.transpose();
        // Hess_pos += norm_vec*norm_vec.transpose();
        // std::cout << "Hello H 0.4" << std::endl; 
        double dt = laser_p.curvature/double(1000);
        auto dR = Exp(s.omg, dt);
        M3D point_dR_crossmat;
        // V3D tmp = ;
        point_dR_crossmat<<SKEW_SYM_MATRX((dR*point_this));
        // if (dt > max_dt) max_dt = dt;

        ekfom_data.h_x.block<1, 3>(i,3) = -weight*norm_vec.transpose()*s.rot.toRotationMatrix()*point_dR_crossmat;
        ekfom_data.h_x.block<1, 3>(i,6) = weight*norm_vec.transpose()*dt;
        ekfom_data.h_x.block<1, 3>(i,15) = -weight*norm_vec.transpose()*s.rot.toRotationMatrix()*dR*point_crossmat*dt;
        // ekfom_data.h_x.block<1, 3>(i,15) = -weight*norm_vec.transpose()*s.rot.toRotationMatrix()*point_crossmat*dt;
        // point_rot_crossmat*dt;
        ekfom_data.h_x.block<1, 3>(i,18) = 0.5*weight*norm_vec.transpose()*dt*dt;
        if (extrinsic_est_en)
        {
            ekfom_data.h_x.block<1, 3>(i,9) = -weight*norm_vec.transpose()*s.rot.toRotationMatrix()*dR*s.offset_R_L_I.toRotationMatrix()*point_be_crossmat;
            ekfom_data.h_x.block<1, 3>(i,12) = weight*norm_vec.transpose()*s.rot.toRotationMatrix()*dR;
        }
        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -weight*norm_p.intensity;
    }  
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Hess_pos);
    // printf("%f %f %f \n", saes.eigenvalues()[0], saes.eigenvalues()[1], saes.eigenvalues()[2]);  

    Eigen::Quaterniond quat_cur(s.rot_cur.toRotationMatrix());
    Eigen::Quaterniond quat(s.rot.toRotationMatrix());
    Eigen::Quaterniond d_quat(Exp(s.omg, lidar_mean_scantime));
    Eigen::Quaterniond res_quat = (quat*d_quat).conjugate()*quat_cur;
    ekfom_data.h.block<3, 1>(effct_feat_num,0) = 2*res_quat.vec();
    ekfom_data.h_x.block<3, 3>(effct_feat_num,24) = -(Qleft(res_quat)).bottomRightCorner<3, 3>();
    ekfom_data.h_x.block<3, 3>(effct_feat_num,3) = (Qright(quat.conjugate()*quat_cur)*Qleft(d_quat.conjugate())).bottomRightCorner<3, 3>();
    ekfom_data.h_x.block<3, 3>(effct_feat_num,15) = lidar_mean_scantime*(Qright((quat*d_quat).conjugate()*quat_cur)).bottomRightCorner<3, 3>();
    
    ekfom_data.h.block<3, 1>(effct_feat_num+3,0) = (s.pos_cur - s.pos - s.vel*lidar_mean_scantime - 0.5*s.acc*lidar_mean_scantime*lidar_mean_scantime);
    ekfom_data.h_x.block<3, 3>(effct_feat_num+3,21) = -Eigen::Matrix3d::Identity();
    ekfom_data.h_x.block<3, 3>(effct_feat_num+3,0) = Eigen::Matrix3d::Identity();
    ekfom_data.h_x.block<3, 3>(effct_feat_num+3,6) = Eigen::Matrix3d::Identity()*lidar_mean_scantime;
    ekfom_data.h_x.block<3, 3>(effct_feat_num+3,18) = 0.5*Eigen::Matrix3d::Identity()*lidar_mean_scantime*lidar_mean_scantime;

    ekfom_data.h.block<3, 1>(effct_feat_num+6,0) = (s.vel_cur - s.vel - s.acc*lidar_mean_scantime);
    ekfom_data.h_x.block<3, 3>(effct_feat_num+6,27) = -Eigen::Matrix3d::Identity();
    ekfom_data.h_x.block<3, 3>(effct_feat_num+6,6) = Eigen::Matrix3d::Identity();
    ekfom_data.h_x.block<3, 3>(effct_feat_num+6,18) = Eigen::Matrix3d::Identity()*lidar_mean_scantime;

    Eigen::Quaternion<double> q = s.rot;
    ekfom_data.h.block<3, 1>(effct_feat_num+9,0) = 2*(last_Q_cur.conjugate()*q).vec();
    ekfom_data.h_x.block<3, 3>(effct_feat_num+9,3) = -(Qright(q.conjugate()*last_Q_cur)).bottomRightCorner<3, 3>();

    ekfom_data.h.block<3, 1>(effct_feat_num+12,0) = (last_P_cur - s.pos);
    ekfom_data.h_x.block<3, 3>(effct_feat_num+12,0) = Eigen::Matrix3d::Identity();

    ekfom_data.h.block<3, 1>(effct_feat_num+15,0) = (last_V_cur - s.vel);
    ekfom_data.h_x.block<3, 3>(effct_feat_num+15,6) = Eigen::Matrix3d::Identity();

    solve_time += omp_get_wtime() - solve_start_;
}

void lioThread() {
    lidar_mean_scantime = 1.0/fix_rate;
    ros::Rate rate(fix_rate);
    bool status = ros::ok();
    while (status) {
        if (flg_exit) {
            // std::cout << "Hello 0" << std::endl;
            break;
        }
        mtx_buffer.lock();
        bool sync = sync_packages(Measures);
        mtx_buffer.unlock();
        if(sync) {
            if (flg_first_scan) {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                // mtx_buffer.unlock();
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            // cout << Measures.lidar.size() << " " << feats_undistort->size() << endl;
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                // mtx_buffer.unlock();
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            // std::cout << Measures.lidar_beg_time - first_lidar_time << " " << flg_EKF_inited << std::endl;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
#ifdef USE_voxel
            if(!surf_map.size() || !flg_EKF_inited) {
                if(feats_down_size > 5) {
                    feats_down_world->resize(feats_down_size);
                    cut_voxel(surf_map, feats_down_body, state_point);
                }
                // std::cout << surf_map.size() << std::endl;
                // mtx_buffer.unlock();
                continue;
            }
#else 
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr) {
                if(feats_down_size > 5) {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    ikdtree.Build(feats_down_world->points);
                }
                // mtx_buffer.unlock();
                continue;
            }
            // int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
#endif            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("Feature points < 5, skip this scan!\n");
                // mtx_buffer.unlock();
                continue;
            }

            
            // double time_hash = omp_get_wtime();
            // unordered_map<VOXEL_LOC, OCTO_TREE*> cloud_map;            
            // std::for_each(std::execution::unseq, feats_down_body->points.begin(), feats_down_body->points.end(), [&cloud_map](const auto& pt) {
            //     V3D pvec_tran(pt.x, pt.y, pt.z);
            //     float loc_xyz[3];
            //     for(int j=0; j<3; j++) {
            //         loc_xyz[j] = pvec_tran[j];
            //         if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;        
            //     }
            //     VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
            //     // Find corresponding voxel
            //     auto iter = cloud_map.find(position);//找到与当前特征对应的体素
            //     if(iter != cloud_map.end()) {
            //         iter->second->plvec_tran->push_back(pvec_tran);
            //     }
            //     else {// If not finding, build a new voxel        
            //         OCTO_TREE *ot = new OCTO_TREE();    //建立一个新体素
            //         // ot->tmp_ori[frame_id].push_back(pvec_orig);  // shm: add the original point into the silding window
            //         ot->plvec_tran->push_back(pvec_tran);       //点云坐标(world fixed frame)
            //         // Voxel center coordinate
            //         ot->voxel_center[0] = (0.5+position.x);
            //         ot->voxel_center[1] = (0.5+position.y);
            //         ot->voxel_center[2] = (0.5+position.z);
            //         ot->quater_length = 0.25; // A quater of side length
            //         cloud_map[position] = ot;
            //     }
            // });
            // std::for_each(std::execution::par_unseq, cloud_map.begin(), cloud_map.end(), [](const auto& hash_map) {
            //     // hash_map.second->feat_eigen_limit = 4;
            //     hash_map.second->recut(0, 1, hash_map.second->root_centors, hash_map.second->pl_eigen, 5);
            // });            
            // // double time_exe = omp_get_wtime();
            // std::for_each(std::execution::par_unseq, feats_down_body->points.begin(), feats_down_body->points.end(), [&cloud_map](auto& pt) {
            //     V3D pvec_tran(pt.x, pt.y, pt.z);
            //     float loc_xyz[3];
            //     for(int j=0; j<3; j++) {
            //         loc_xyz[j] = pvec_tran[j];
            //         if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;        
            //     }
            //     VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
            //     // Find corresponding voxel
            //     auto iter = cloud_map.find(position);//找到与当前特征对应的体素
            //     if(iter != cloud_map.end() && iter->second->root_centors.size()) {
            //         Eigen::Vector3d C = Eigen::Vector3d(iter->second->root_centors[0].normal_x, iter->second->root_centors[0].normal_y, iter->second->root_centors[0].normal_z);
            //         pt.normal_x = C(0);
            //         pt.normal_y = C(1);
            //         pt.normal_z = C(2);
            //         // Cov += C*C.transpose();
            //     }
            // });
            Eigen::Matrix<double, 3, 3> Cov = Eigen::Matrix<double, 3, 3>::Zero();
            // std::for_each(std::execution::unseq, feats_down_body->points.begin(), feats_down_body->points.end(), [&Cov](auto& pt) {
            //     Eigen::Vector3d C = Eigen::Vector3d(pt.normal_x, pt.normal_y, pt.normal_z);
            //     Cov += C*C.transpose();
            // });
            // time_hash = omp_get_wtime() - time_hash;
            // // std::cout << time_for << " " << time_exe << std::endl;
            double time_pcl = omp_get_wtime();
#ifdef MP_EN
            pcl::NormalEstimationOMP<PointType, pcl::Normal> ne(MP_PROC_NUM);
#else
            pcl::NormalEstimation<PointType, pcl::Normal> ne;
#endif
            ne.setInputCloud(feats_down_body);
            pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
            ne.setSearchMethod (tree);
            pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
            ne.setRadiusSearch (1);
            ne.compute (*cloud_normals);
            // Eigen::Matrix<double, 3, 3> Cov = Eigen::Matrix<double, 3, 3>::Zero();
            for (int kk=0; kk<cloud_normals->size(); kk++) {
                Eigen::Vector3d C = Eigen::Vector3d(cloud_normals->points[kk].normal_x, cloud_normals->points[kk].normal_y, cloud_normals->points[kk].normal_z);
                if (C.allFinite()) {
                    feats_down_body->points[kk].normal_x = cloud_normals->points[kk].normal_x;
                    feats_down_body->points[kk].normal_y = cloud_normals->points[kk].normal_y;
                    feats_down_body->points[kk].normal_z = cloud_normals->points[kk].normal_z;
                    Cov += C*C.transpose();
                }
            }
            // std::cout << omp_get_wtime() - time_pcl << std::endl;
            // time_pcl = omp_get_wtime() - time_pcl;
            // std::cout << time_hash << " " << time_pcl << std::endl;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Cov);
            auto eigenVec = saes.eigenvectors(); 
            std::vector<std::vector<std::pair<int, double>>> index(3);
            for (int kk=0; kk<feats_down_body->points.size(); kk++) {
                Eigen::Vector3d C = Eigen::Vector3d(feats_down_body->points[kk].normal_x, feats_down_body->points[kk].normal_y, feats_down_body->points[kk].normal_z);
                if (C.allFinite()) {
                    std::vector<double> V;
                    for (int m=0; m<3; m++) {
                        V.push_back(abs(C.dot(eigenVec.block<3, 1>(0,m))));
                    }
                    auto maxElement = std::max_element(V.begin(), V.end());
                    if (*maxElement > 0.8) {
                        size_t maxIndex = std::distance(V.begin(), maxElement);
                        index[maxIndex].push_back(std::make_pair(kk,*maxElement));
                    }
                }
            }
            PointCloudXYZI::Ptr feats_rms_body(new PointCloudXYZI());
            for(int m=0; m<3; m++) {
                if(index[m].size()) {
                    sort(index[m].begin(),index[m].end(),norm_list);
                } else continue;
                int point_size = index[m].size()<800? index[m].size():800;
                for(int kk=0; kk<point_size; kk++) {
                    feats_rms_body->points.push_back(feats_down_body->points[index[m][kk].first]);
                }
            }
            // feats_down_size = feats_rms_body->points.size();
            // *feats_down_body = *feats_rms_body;
            
            // std::cout << "**************" << std::endl;
            //  
            // std::cout << omp_get_wtime(  ) - time_hash << std::endl;
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            // std::cout << "Hello -1" << std::endl;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            
            // std::cout << "Hello 0" << std::endl;
            state_point = kf.get_x();
            // auto state_cov = kf.get_P();
            
            
            // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(state_cov.inverse().block<3, 3>(0,0));
            // printf("%f %f %f \n", saes.eigenvalues()[0], saes.eigenvalues()[1], saes.eigenvalues()[2]);

            // std::cout << "Hello 1" << std::endl;
            euler_cur = SO3ToEuler(state_point.rot_cur);
            // std::cout << "Hello 2" << std::endl;
            pos_lid = state_point.pos_cur + state_point.rot_cur * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot_cur.coeffs()[0];
            geoQuat.y = state_point.rot_cur.coeffs()[1];
            geoQuat.z = state_point.rot_cur.coeffs()[2];
            geoQuat.w = state_point.rot_cur.coeffs()[3];

            last_P_cur = state_point.pos_cur;
            last_V_cur = state_point.vel_cur;
            Eigen::Quaternion<double> quat_cur(state_point.rot_cur.toRotationMatrix());
            last_Q_cur = quat_cur;

            double t_update_end = omp_get_wtime();
            // std::cout << "Hello 3" << std::endl;
            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
#ifdef USE_voxel
            cut_voxel(surf_map, feats_down_body, state_point);
#else
            map_incremental();
#endif
            // std::cout << "Hello 5" << std::endl;
            t5 = omp_get_wtime();

            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log) {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }
        // mtx_buffer.unlock();
        status = ros::ok();
        rate.sleep();
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("mapping/offline_map_path",offline_map_path,"");
    nh.param<int>("common/lidar_num", lidar_num, 1);
    std::vector<std::string> lid_topic(lidar_num);
    nh.param<std::vector<std::string>>("common/lid_topic",lid_topic,std::vector<std::string>());
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<string>("common/imu_odom_topic",imu_odom_topic,"/Odometry/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<double>("mapping/root_surf_voxel_size",rootSurfVoxelSize,1.0);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    // nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    // nh.param<double>("mapping/lidar_mean_scantime",lidar_mean_scantime,0.01);
    nh.param<int>("mapping/rate", fix_rate, 50);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<std::vector<int>>("preprocess/lidar_type", p_pre->lidar_type, std::vector<int>());
    nh.param<std::vector<int>>("preprocess/scan_line", p_pre->N_SCANS, std::vector<int>());
    nh.param<std::vector<int>>("preprocess/timestamp_unit", p_pre->time_unit, std::vector<int>());
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("mapping/use_kernal", use_kernal, false);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<bool>("mapping/load_offline_map", load_offline_map, false);
    // cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0;
    double deltaT, deltaR;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    // FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    // HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    // memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));



    double epsi[38] = {0.001};
    fill(epsi, epsi+38, 0.001);
    // std::cout << "Hello 0" << std::endl;
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);
    
    /*** debug record ***/
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");


    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    lidar_msg_buffer.resize(lidar_num);
    Measures.lidar.resize(lidar_num);
    std::vector<ros::Subscriber> sub_pcl(lidar_num);
    for (int num = 0; num<lidar_num; num++) {
        Eigen::Vector3d extrinsic_Tn(extrinT.at(num * 3), extrinT.at(num * 3 + 1), extrinT.at(num * 3 + 2));
        Textrinsic.push_back(extrinsic_Tn);
        Eigen::Quaterniond extrinsic_Qn(extrinR.at(num * 4), extrinR.at(num * 4 + 1), extrinR.at(num * 4 + 2), extrinR.at(num * 4 + 3)); //w,x,y,z
        Rextrinsic.push_back(extrinsic_Qn.toRotationMatrix());
        if (p_pre->lidar_type[num]==ROBOSENSE) 
            sub_pcl[num] = nh.subscribe<sensor_msgs::PointCloud2>(lid_topic.at(num), 1,
                                [num](const sensor_msgs::PointCloud2::ConstPtr& msg) {
                                    psa_rs_pcl_cbk(msg, num);
                                });
        else
            sub_pcl[num] = nh.subscribe<sensor_msgs::PointCloud2>(lid_topic.at(num), 1,
                                [num](const sensor_msgs::PointCloud2::ConstPtr& msg) {
                                    standard_pcl_cbk(msg, num);
                                });
    }

    // Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    // Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    // p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    // Eigen::Quaterniond quat_cur(Lidar_R_wrt_IMU);
    // printf("Quat: %f %f %f %f\n",quat_cur.w(), quat_cur.x(), quat_cur.y(), quat_cur.z());
    p_imu->set_extrinsic(Textrinsic[0], Rextrinsic[0]);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    
    /*** ROS subscribe initialization ***/
    // ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
    //     nh.subscribe(lid_topic[0], 1, livox_pcl_cbk) : \
    //     nh.subscribe(lid_topic[0], 1, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 1, imu_cbk);
    pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2> ("/cloud_registered", 100000);
    pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2> ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2> ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2> ("/Laser_map", 100000);
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/Odometry", 1);
    ros::Publisher pubOffilineMap = nh.advertise<sensor_msgs::PointCloud2> ("/offline_cloud", 1);
    pubOdomImu = nh.advertise<nav_msgs::Odometry> (imu_odom_topic, 1);
    pubLaserCloudSeg = nh.advertise<sensor_msgs::PointCloud2> ("/Laser_seg", 100000);
    pubPath = nh.advertise<nav_msgs::Path> ("/path", 100000);
//------------------------------------------------------------------------------------------------------
    if (load_offline_map) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
        string loadMapDirectory;
        loadMapDirectory = std::getenv("HOME");// + "work/ctlimo_ws/src/FAST_LIO/PCD";
        // cout << "Load destination: " << loadMapDirectory << endl;
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(loadMapDirectory + offline_map_path, *cloud_pcd) == -1) {
            PCL_ERROR("Couldn't read PCD file.\n");
        }
        std::cout << "Loaded " << cloud_pcd->width * cloud_pcd->height << " data points from PCD file." << std::endl;
        ros::Rate rate_pcd(0.5);
        rate_pcd.sleep();
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*cloud_pcd, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time::now();
        laserCloudmsg.header.frame_id = "camera_init";
        pubOffilineMap.publish(laserCloudmsg);
    }

    signal(SIGINT, SigHandle);
    
    // std::cout << 1/lidar_mean_scantime << std::endl;
    // int rate_ = 1/lidar_mean_scantime;
    std::thread liothread(lioThread);
    // std::thread testthread(testThread);
    ros::MultiThreadedSpinner spinner(lidar_num+1);
    spinner.spin();
    liothread.join();
    // testthread.join();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
