#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "common_lib.hpp"

using namespace std;

#define IS_VALID(a)  ((abs(a)>1e8) ? true : false)
#define MIN_PS 7


// namespace mlio {
//   struct EIGEN_ALIGN16 Point {
//       PCL_ADD_POINT4D;
//       float intensity;
//       float normal_x;
//       float normal_y;
//       float normal_z;
//       float curvature;
//       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//   };
// }  // namespace velodyne_ros
// POINT_CLOUD_REGISTER_POINT_STRUCT(mlio::Point,
//     (float, x, x)
//     (float, y, y)
//     (float, z, z)
//     (float, intensity, intensity)
//     (float, normal_x, normal_x)
//     (float, normal_y, normal_y)
//     (float, normal_z, normal_z)
//     (float, curvature, curvature)
// )
// using PointType = mlio::Point;
// typedef mlio::Point PointType1;
// typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE{AVIA = 1, VELO16, OUST64, ROBOSENSE}; //{1, 2, 3}
enum TIME_UNIT{SEC = 0, MS = 1, US = 2, NS = 3};
enum Feature{Nor, Poss_Plane, Real_Plane, Edge_Jump, Edge_Plane, Wire, ZeroPoint};
enum Surround{Prev, Next};
enum E_jump{Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind};

struct orgtype
{
  double range;
  double dista; 
  double angle[2];
  double intersect;
  E_jump edj[2];
  Feature ftype;
  orgtype()
  {
    range = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype = Nor;
    intersect = 2;
  }
};

namespace velodyne_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      double time;
      uint16_t ring;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (double, time, time)
    (uint16_t, ring, ring)
)

namespace robosense_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      uint32_t sec;
      uint32_t usec;
      uint16_t ring;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace robosense_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(robosense_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint32_t, sec, sec)
    (uint32_t, usec, usec)
    (uint16_t, ring, ring)
)

namespace ouster_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      uint32_t t;
      uint16_t reflectivity;
      // uint8_t  ring;
      uint16_t ambient;
      uint32_t range;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    // (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

class Preprocess
{
  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess() {
    feature_enabled = 0;
    blind = 0.01;
    point_filter_num = 1;
    inf_bound = 10;
    // N_SCANS   = 6;
    SCAN_RATE = 10;
    group_size = 8;
    disA = 0.01;
    disA = 0.1; // B?
    p2l_ratio = 225;
    limit_maxmid =6.25;
    limit_midmin =6.25;
    limit_maxmin = 3.24;
    jump_up_limit = 170.0;
    jump_down_limit = 8.0;
    cos160 = 160.0;
    edgea = 2;
    edgeb = 0.1;
    smallp_intersect = 172.5;
    smallp_ratio = 1.2;
    given_offset_time = false;

    jump_up_limit = cos(jump_up_limit/180*M_PI);
    jump_down_limit = cos(jump_down_limit/180*M_PI);
    cos160 = cos(cos160/180*M_PI);
    smallp_intersect = cos(smallp_intersect/180*M_PI);
  }
  ~Preprocess(){}
  
  // void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out, const int lidar_id);
  void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out, const int lidar_id) {
    switch (time_unit[lidar_id])
    {
      case SEC:
        time_unit_scale = 1.e3f;
        break;
      case MS:
        time_unit_scale = 1.f;
        break;
      case US:
        time_unit_scale = 1.e-3f;
        break;
      case NS:
        time_unit_scale = 1.e-6f;
        break;
      default:
        time_unit_scale = 1.f;
        break;
    }
    // std::cout << lidar_type[lidar_id] << " " << ROBOSENSE << std::endl;
    switch (lidar_type[lidar_id])
    {
    case OUST64:
      oust64_handler(msg, lidar_id);
      break;

    case VELO16:
      velodyne_handler(msg, lidar_id);
      break;

    case ROBOSENSE:
      robosense_handler(msg, lidar_id);
      break;
    
    default:
      printf("Error LiDAR Type");
      break;
    }
    *pcl_out = pl_surf;
  }
  // void set(bool feat_en, int lid_type, double bld, int pfilt_num);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI pl_surf;
  PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
  vector<orgtype> typess[128]; //maximum 128 line lidar
  float time_unit_scale;
  int point_filter_num, SCAN_RATE;
  std::vector<int> lidar_type, N_SCANS, time_unit;
  double blind;
  bool feature_enabled, given_offset_time;
  ros::Publisher pub_full, pub_surf, pub_corn;
    

  private:
  // void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg, const int lidar_id);
  void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, const int lidar_id) {
    pl_surf.clear();
    // pl_corn.clear();
    // pl_full.clear();
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    // pl_corn.reserve(plsize);
    pl_surf.reserve(plsize);

    double time_stamp = msg->header.stamp.toSec();
    // cout << "===================================" << endl;
    // printf("Pt size = %d, N_SCANS = %d\r\n", plsize, N_SCANS);
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
      if (i % point_filter_num != 0) continue;

      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      
      if (range < (blind * blind)) continue;
      
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.curvature = pl_orig.points[i].t * time_unit_scale; // curvature unit: ms
      // std::cout << added_pt.curvature << std::endl;
      pl_surf.points.push_back(added_pt);
    }
    // pub_func(pl_surf, pub_full, msg->header.stamp);
    // pub_func(pl_surf, pub_corn, msg->header.stamp);
  }
  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, const int lidar_id) {
    pl_surf.clear();
    // pl_corn.clear();
    // pl_full.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    pl_surf.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * SCAN_RATE;       // scan angular velocity, curvature unit: ms
    std::vector<bool> is_first(N_SCANS[lidar_id],true);
    std::vector<double> yaw_fp(N_SCANS[lidar_id], 0.0);      // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS[lidar_id], 0.0);   // yaw of last scan point
    std::vector<float> time_last(N_SCANS[lidar_id], 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0)
    {
      given_offset_time = true;
    }
    else
    {
      given_offset_time = false;
      double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
      double yaw_end  = yaw_first;
      int layer_first = pl_orig.points[0].ring;
      for (uint i = plsize - 1; i > 0; i--)
      {
        if (pl_orig.points[i].ring == layer_first)
        {
          yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
          break;
        }
      }
    }
    
    for (int i = 0; i < plsize; i++)
    {
      PointType added_pt;
      // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;
      
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.curvature = pl_orig.points[i].time * time_unit_scale;  // curvature unit: ms

      if (!given_offset_time)
      {
        int layer = pl_orig.points[i].ring;
        double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

        if (is_first[layer])
        {
          // printf("layer: %d; is first: %d", layer, is_first[layer]);
            yaw_fp[layer]=yaw_angle;
            is_first[layer]=false;
            added_pt.curvature = 0.0;
            yaw_last[layer]=yaw_angle;
            time_last[layer]=added_pt.curvature;
            continue;
        }

        // compute offset time
        if (yaw_angle <= yaw_fp[layer])
        {
          added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
        }
        else
        {
          added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
        }

        // if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

        yaw_last[layer] = yaw_angle;
        time_last[layer]=added_pt.curvature;
      }

      if (i % point_filter_num == 0)
      {
        if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
        {
          pl_surf.points.push_back(added_pt);
        }
      }
    }
  }
  void robosense_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, const int lidar_id) {
    pl_surf.clear();
    // pl_corn.clear();
    // pl_full.clear();
    pcl::PointCloud<robosense_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    // pl_corn.reserve(plsize);
    pl_surf.reserve(plsize);

    double time_stamp = msg->header.stamp.toSec();
    // cout << "===================================" << endl;
    // printf("Pt size = %d, N_SCANS = %d\r\n", plsize, N_SCANS);
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
      if (i % point_filter_num != 0) continue;

      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      
      if (range < (blind * blind)) continue;
      
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.curvature = ((pl_orig.points[i].sec + pl_orig.points[i].usec*1e-6)-time_stamp)*1e3; //ms
      // printf("%.6f\n", added_pt.curvature);
      pl_surf.points.push_back(added_pt);
    }
    // pub_func(pl_surf, pub_full, msg->header.stamp);
    // pub_func(pl_surf, pub_corn, msg->header.stamp);
  }
  // void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
  // void pub_func(PointCloudXYZI &pl, const ros::Time &ct);
  // int  plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  // bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct);
  // bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);
  
  int group_size;
  double disA, disB, inf_bound;
  double limit_maxmid, limit_midmin, limit_maxmin;
  double p2l_ratio;
  double jump_up_limit, jump_down_limit;
  double cos160;
  double edgea, edgeb;
  double smallp_intersect, smallp_ratio;
  double vx, vy, vz;
};

/* Key of Hash Table */
class VOXEL_LOC
{
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx=0, int64_t vy=0, int64_t vz=0): x(vx), y(vy), z(vz){}

  bool operator== (const VOXEL_LOC &other) const
  {
    return (x==other.x && y==other.y && z==other.z);
  }
};

/* Hash value */
namespace std
{
  template<>
  struct hash<VOXEL_LOC>
  {
    size_t operator() (const VOXEL_LOC &s) const
    {
      using std::size_t; using std::hash;
      return ((hash<int64_t>()(s.x) ^ (hash<int64_t>()(s.y) << 1)) >> 1) ^ (hash<int64_t>()(s.z) << 1);
    }
  };
}

class OCTO_TREE
{
public:
  std::vector<Eigen::Vector3d>* plvec_tran;
  bool octo_state; // 0 is end of tree, 1 is not
  // int ftype;//选择线或者面
  double feat_eigen_ratio;
  // PointType ap_centor_direct;
  // PointXYZCOV ap_centor_cov;
  double voxel_center[3]; // x, y, z
  double quater_length;
  OCTO_TREE* leaves[8];
  bool is2opt;// correspond;
  pcl::PointCloud<PointType> root_centors;
  // pcl::PointCloud<PointXYZCOV> root_centors_cov;
  Eigen::Vector3d ap_enigen;
  std::vector<Eigen::Vector3d> pl_eigen;
  // std::vector<std::vector<Eigen::Vector3d>> tmp_ori;
  double Time;
  // double correspondTime;
  float feat_eigen_limit = 9;

  OCTO_TREE()
  {    
    // tmp_ori.resize(WINDOW_SIZE+1);
    // for(int i=0; i<WINDOW_SIZE+1; i++) tmp_ori[i].clear();
    octo_state = false;
    for(int i=0; i<8; i++) leaves[i] = nullptr;
    plvec_tran = new std::vector<Eigen::Vector3d>();
    is2opt = true;
    pl_eigen.clear();
    // correspond = false;
    Time = ros::Time::now().toSec();
  }

  // Used by "recut"
  PointType calc_eigen()
  {
    Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
    Eigen::Vector3d center(0, 0, 0);

    for(uint j=0; j<plvec_tran->size(); j++)
    {
      covMat += (*plvec_tran)[j] * (*plvec_tran)[j].transpose();
      center += (*plvec_tran)[j];
    }

    center /= plvec_tran->size();

    covMat = covMat/plvec_tran->size() - center*center.transpose();
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
    feat_eigen_ratio = saes.eigenvalues()[2] / saes.eigenvalues()[0];
    Eigen::Vector3d direct_vec = saes.eigenvectors().col(0);
    
    ap_enigen << saes.eigenvalues()[0], saes.eigenvalues()[1], saes.eigenvalues()[2];
    PointType ap_centor_direct;
    ap_centor_direct.x = center.x();
    ap_centor_direct.y = center.y();
    ap_centor_direct.z = center.z();
    ap_centor_direct.normal_x = direct_vec.x();
    ap_centor_direct.normal_y = direct_vec.y();
    ap_centor_direct.normal_z = direct_vec.z();
    return ap_centor_direct;
  }

  // Cut root voxel into small pieces
  void recut(int layer, int layer_max, pcl::PointCloud<PointType> &pl_feat_map, std::vector<Eigen::Vector3d> &pl_feat_eigen, 
            const int min_size=MIN_PS) {
    if(plvec_tran->size() < min_size) {
      feat_eigen_ratio = -1;
      if(layer != 0 && plvec_tran != nullptr) std::vector<Eigen::Vector3d>().swap(*plvec_tran);
      return;
    }

    PointType ap_centor_direct = calc_eigen(); // calculate eigenvalue ratio
    
    if(std::isnan(feat_eigen_ratio)) {
      feat_eigen_ratio = -1;
      if(layer != 0 && plvec_tran != nullptr) std::vector<Eigen::Vector3d>().swap(*plvec_tran);
      return;
    }

    if(feat_eigen_ratio >= feat_eigen_limit) {
      pl_feat_map.push_back(ap_centor_direct);
      pl_feat_eigen.push_back(ap_enigen);
      // pl_feat_map1.push_back(ap_centor_cov);
      if(layer != 0 && plvec_tran != nullptr) std::vector<Eigen::Vector3d>().swap(*plvec_tran);
      return;
    }

    if(layer == layer_max) {
      if(plvec_tran != nullptr) std::vector<Eigen::Vector3d>().swap(*plvec_tran);
      return;
    }

    int leafnum;

    for(uint j=0; j<plvec_tran->size(); j++) {
      int xyz[3] = {0, 0, 0};
      for(uint k=0; k<3; k++) {
        if((*plvec_tran)[j][k] > voxel_center[k]) xyz[k] = 1;
      }
      leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr) {
        leaves[leafnum] = new OCTO_TREE();
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }
      leaves[leafnum]->plvec_tran->push_back((*plvec_tran)[j]);
    }
    
    if(layer != 0 && plvec_tran != nullptr) std::vector<Eigen::Vector3d>().swap(*plvec_tran);

    layer++;
    for(uint i=0; i<8; i++) {
      if(leaves[i] != nullptr) leaves[i]->recut(layer, layer_max, pl_feat_map, pl_feat_eigen, min_size);//, pl_feat_map1);
    }
  }
};
#endif