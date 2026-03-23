#include <iostream>
#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>

class ReLocalization {
public:
    ReLocalization() : 
        has_initial_pose_(false), initial_pose_(Eigen::Matrix4f::Identity()) {
        // map_origin_ptr_.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
    }

    ~ReLocalization() = default;

    void GetInitialPose(const geometry_msgs::PoseStamped::ConstPtr &goal) {
        initial_pose_(0,3) = goal->pose.position.x;
        initial_pose_(1,3) = goal->pose.position.y;
        initial_pose_(2,3) = goal->pose.position.z;

        tf::Quaternion q(
            goal->pose.orientation.x,
            goal->pose.orientation.y,
            goal->pose.orientation.z,
            goal->pose.orientation.w
        );

        tf::Matrix3x3 R(q);

        initial_pose_(0, 0) = R[0][0];
        initial_pose_(0, 1) = R[0][1];
        initial_pose_(0, 2) = R[0][2];
        initial_pose_(1, 0) = R[1][0];
        initial_pose_(1, 1) = R[1][1];
        initial_pose_(1, 2) = R[1][2];
        initial_pose_(2, 0) = R[2][0];
        initial_pose_(2, 1) = R[2][1];
        initial_pose_(2, 2) = R[2][2];
        // goal->pose.orientation.quaternion.
        has_initial_pose_ = true;
        std::cout << "Get initial pose from rivz =\n " << initial_pose_ << std::endl;
    }

    bool LoadMap(std::string map_path_, pcl::PointCloud<pcl::PointXYZINormal>& map) {

        // std::lock_guard<std::mutex> lock(mtx_);

        if (pcl::io::loadPCDFile<pcl::PointXYZINormal> (map_path_, map) == -1){
            std::cout << "### Couldn't read file scans.pcd \n" << std::endl;
            return false;
        }

        //! too werid for if()
        pcl::io::loadPCDFile<pcl::PointXYZINormal> (map_path_, map);
        std::cout << "The map size = " << map.points.size() << std::endl;
        pcl::VoxelGrid<pcl::PointXYZINormal> pt2_filter;
        pt2_filter.setInputCloud(map.makeShared());
        pt2_filter.setLeafSize(kMAPVOXELSIZE, kMAPVOXELSIZE, kMAPVOXELSIZE);
        pt2_filter.filter(map);

        if (map.points.size() == 0) return false;
            

        pcl::VoxelGrid<pcl::PointXYZINormal> pt_filter;
        pt_filter.setInputCloud(map.makeShared());
        pt_filter.setLeafSize(kMAPVOXELSIZE, kMAPVOXELSIZE, kMAPVOXELSIZE);
        pt_filter.filter(map);
        
        return true;
    }

    bool has_initial_pose_;
    Eigen::Matrix4f initial_pose_;
private:
    // std::string map_path_ = "";    
    const float kMAPVOXELSIZE = 0.2;
    // pcl::PointCloud<pcl::PointXYZINormal>::Ptr map_origin_ptr_;
    // std::mutex mtx_;
};