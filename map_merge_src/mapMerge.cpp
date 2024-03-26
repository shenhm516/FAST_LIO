#include <mutex>
// #include <math.h>
// #include <thread>
// #include <fstream>
// #include <csignal>
// #include <unistd.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/gicp.h>
#include <sensor_msgs/PointCloud2.h>
#include "so3_math.h"

struct PointXYZISE3T
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float rx;
    float ry;
    float rz;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZISE3T,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, rx, rx) (float, ry, ry) (float, rz, rz)
                                   (double, time, time))

std::string loadMapDirectory, base_map_file_path, source_map_file_path;
pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterICP;
std::deque<pcl::PointCloud<pcl::PointXYZI>> lidar_buffer;
// std::mutex mtx_buffer;
pcl::PointCloud<pcl::PointXYZI>::Ptr base_cloud;
// SCManager scManager;
std::vector<ros::Publisher> pubOffilineMapSource;
// pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
// std::vector<pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>> icp;
std::vector<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI>> icp;
std::vector<Eigen::Vector3d> Textrinsic;
std::vector<Eigen::Matrix<double,3,3>> Rextrinsic;
std::vector<Eigen::Affine3f> affineMatrix;
std::vector<bool> first_transform;
std::vector<bool> add_point;
int lidar_num = 1;
std::vector<double> icp_score;

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg, const int& lidar_id) 
{
    pcl::PointCloud<pcl::PointXYZI> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);    

    pcl::PointCloud<pcl::PointXYZI> pl_global;
    pl_global.resize(pl_orig.size());
    // init transform
    if (first_transform[lidar_id]) {
        affineMatrix[lidar_id] = Eigen::Translation3f(Textrinsic[lidar_id].cast<float>())*Rextrinsic[lidar_id].cast<float>();
        first_transform[lidar_id] = false;
    }
    pcl::transformPointCloud(pl_orig, pl_global, affineMatrix[lidar_id]); 

    // Align clouds
    // mtx_buffer.lock();
    icp[lidar_id].setInputSource(pl_global.makeShared());
    pcl::PointCloud<pcl::PointXYZI>::Ptr unused_result(new pcl::PointCloud<pcl::PointXYZI>());
    icp[lidar_id].align(*unused_result);
    // std::cout << icp[lidar_id].getFitnessScore() << " " << lidar_id << std::endl;

    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp[lidar_id].getFinalTransformation();
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);    
    affineMatrix[lidar_id] = correctionLidarFrame*affineMatrix[lidar_id];
    pcl::transformPointCloud(pl_orig, *transformed_cloud, affineMatrix[lidar_id]);
    
    // if(icp[lidar_id].getFitnessScore() < 0.12 && add_point[lidar_id]) {
    double score = icp[lidar_id].getFitnessScore();
    if(score < icp_score[lidar_id]) {
        icp_score[lidar_id] = score;
        std::string all_points_dir = loadMapDirectory + "psacalibr_" + std::to_string(lidar_id) + std::string(".pcd");
        pcl::PCDWriter pcd_writer;
        std::cout << "current scan saved to " << all_points_dir << std::endl;
        pcd_writer.writeBinary(all_points_dir, pl_orig);
        PointXYZISE3T pose;
        pose.x = affineMatrix[lidar_id].translation()(0);
        pose.y = affineMatrix[lidar_id].translation()(1);
        pose.z = affineMatrix[lidar_id].translation()(2);
        Eigen::Vector3f rot_ang(Log(affineMatrix[lidar_id].rotation()));
        pose.rx = rot_ang(0);
        pose.ry = rot_ang(1);
        pose.rz = rot_ang(2);
        pcl::PointCloud<PointXYZISE3T>::Ptr posePCD(new pcl::PointCloud<PointXYZISE3T>());
        posePCD->push_back(pose); 
        std::string all_pose_dir = loadMapDirectory + "psacalibr_pose_" + std::to_string(lidar_id) + std::string(".pcd");
        std::cout << "current scan saved to " << all_pose_dir << std::endl;
        pcd_writer.writeBinary(all_pose_dir, *posePCD);
        add_point[lidar_id] = false;
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*transformed_cloud, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time::now();
        laserCloudmsg.header.frame_id = "camera_init";
        pubOffilineMapSource[lidar_id].publish(laserCloudmsg);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    nh.param<std::string>("base_map_file_path", base_map_file_path, "");
    // nh.param<string>("source_map_file_path", source_map_file_path, "");
    downSizeFilterICP.setLeafSize(0.3, 0.3, 0.3);

    double icp_score_the;
    nh.param<double>("icp_score_the", icp_score_the, 0.2);
    base_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    loadMapDirectory = std::string(std::string(ROOT_DIR) + "PCD/");
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(loadMapDirectory + base_map_file_path, *base_cloud) == -1) {
        PCL_ERROR("Couldn't read PCD file.\n");
    }
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZI>());
    downSizeFilterICP.setInputCloud(base_cloud);
    downSizeFilterICP.filter(*cloud_temp);
    *base_cloud = *cloud_temp; 
    // std::cout << loadMapDirectory + base_map_file_path << std::endl;
    std::cout << "Loaded " << base_cloud->width * base_cloud->height << " data points from PCD file." << std::endl;
    
    std::vector<double> extrinT, extrinR;
    nh.param<int>("lidar_num", lidar_num, 1);
    std::vector<ros::Subscriber> sub_pcl(lidar_num);
    pubOffilineMapSource.resize(lidar_num);
    icp.resize(lidar_num);
    affineMatrix.resize(lidar_num);
    first_transform.resize(lidar_num);
    add_point.resize(lidar_num);
    icp_score.resize(lidar_num);
    std::fill(first_transform.begin(), first_transform.end(), true);
    std::fill(add_point.begin(), add_point.end(), true);
    std::fill(icp_score.begin(), icp_score.end(), icp_score_the);

    nh.param<std::vector<double>>("extrinsic_T", extrinT, std::vector<double>());
    nh.param<std::vector<double>>("extrinsic_R", extrinR, std::vector<double>());
    std::vector<std::string> lid_topic;
    nh.param<std::vector<std::string>>("lidar_topic", lid_topic, std::vector<std::string>());
    for (int num = 0; num<lidar_num; num++) {
        Eigen::Vector3d extrinsic_Tn(extrinT.at(num * 3), extrinT.at(num * 3 + 1), extrinT.at(num * 3 + 2));
        Textrinsic.push_back(extrinsic_Tn);
        Eigen::Quaterniond extrinsic_Qn(extrinR.at(num * 4), extrinR.at(num * 4 + 1), extrinR.at(num * 4 + 2), extrinR.at(num * 4 + 3)); //w,x,y,z
        Rextrinsic.push_back(extrinsic_Qn.toRotationMatrix());
        sub_pcl[num] = nh.subscribe<sensor_msgs::PointCloud2>(lid_topic.at(num), 1,
                            [num](const sensor_msgs::PointCloud2::ConstPtr& msg) {
                                standard_pcl_cbk(msg, num);
                            });
        std::string source_cloud_topic = "cloud_registered_source" + std::to_string(num);
        pubOffilineMapSource[num] = nh.advertise<sensor_msgs::PointCloud2> (source_cloud_topic, 1);
        icp[num].setMaxCorrespondenceDistance(10);
        icp[num].setMaximumIterations(10);
        icp[num].setTransformationEpsilon(1e-1);
        icp[num].setEuclideanFitnessEpsilon(1e-1);
        icp[num].setRANSACIterations(0);
        icp[num].setInputTarget(base_cloud);
    }

    ros::Publisher pubOffilineMapBase = nh.advertise<sensor_msgs::PointCloud2> ("/cloud_registered_base", 100000);
    
    ros::Rate rate_pcd(0.5);
    rate_pcd.sleep();

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*base_cloud, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubOffilineMapBase.publish(laserCloudmsg);

    ros::MultiThreadedSpinner spinner(8);
    ros::spin();
    return 0;
}
