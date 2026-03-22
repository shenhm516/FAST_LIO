#include <iostream>
#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>

class ReLocalization {
public:
    ReLocalization() : has_initial_pose_(false), initial_pose_(Eigen::Matrix4f::Identity()) {}

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

    bool has_initial_pose_;
    Eigen::Matrix4f initial_pose_;
};