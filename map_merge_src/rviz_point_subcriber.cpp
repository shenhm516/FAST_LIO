#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <Eigen/Core>

// pcl::PointCloud<pcl::PointXYZ> cloud_total;
ros::Publisher pub_centorid;
std::vector<Eigen::Vector4f> centorid_vec;
void point_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg){
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg(*msg,cloud);
  // cloud_total += cloud;



  printf("Check the point cloud with the initial poses.\n");
  printf("If no problem, input '1' to continue, input '2' to caculate Centorid or '0' to exit...\n");
  int a; std::cin >> a; if(a==0) exit(0);
  if(a==1) {
    Eigen::Vector4f centorid;
    pcl::compute3DCentroid(cloud, centorid);
    centorid_vec.push_back(centorid);
  }
  if(a==2) {
    Eigen::Vector4f centorid;
    pcl::compute3DCentroid(cloud, centorid);
    centorid_vec.push_back(centorid);
    pcl::PointXYZ pp;
    for (int i=0; i<centorid_vec.size(); i++) {
      pp.x += centorid_vec[i][0]/centorid_vec.size();
      pp.y += centorid_vec[i][1]/centorid_vec.size();
      pp.z += centorid_vec[i][2]/centorid_vec.size();
    }
    // Eigen::Vector4f centorid;
    // pcl::compute3DCentroid(cloud_total, centorid);
    // std::cout << centorid.transpose() << std::endl;
    pcl::PointCloud<pcl::PointXYZ> cloud_pub_pcl;
    cloud_pub_pcl.push_back(pp);
    sensor_msgs::PointCloud2 cloud_pub_msg;
    pcl::toROSMsg(cloud_pub_pcl,cloud_pub_msg);
    cloud_pub_msg.header = msg->header;
    pub_centorid.publish(cloud_pub_msg);
    printf("%f %f %f\n", pp.x, pp.y, pp.z);
  }
}

int main(int argc, char **argv){
  ros::init(argc, argv, "rviz_point_subscriber");
  ros::NodeHandle nh;
  // cloud_total.reset(new pcl::PointCloud<pcl::PointXYZ>());
  
  ros::Subscriber point_sub = nh.subscribe("/rviz_selected_points", 10, point_cbk);
  pub_centorid = nh.advertise<sensor_msgs::PointCloud2>("/centorid_point", 100);
  ros::spin();
  return 0;
}