#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using namespace std;

#define IS_VALID(a)  ((abs(a)>1e8) ? true : false)
#define MIN_PS 7

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE{AVIA = 1, VELO16, OUST64}; //{1, 2, 3}
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

namespace ouster_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      uint32_t t;
      uint16_t reflectivity;
      uint8_t  ring;
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
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

class Preprocess
{
  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess();
  ~Preprocess();
  
  void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void set(bool feat_en, int lid_type, double bld, int pfilt_num);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI pl_full, pl_corn, pl_surf;
  PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
  vector<orgtype> typess[128]; //maximum 128 line lidar
  float time_unit_scale;
  int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit;
  double blind;
  bool feature_enabled, given_offset_time;
  ros::Publisher pub_full, pub_surf, pub_corn;
    

  private:
  void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg);
  void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
  void pub_func(PointCloudXYZI &pl, const ros::Time &ct);
  int  plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);
  
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
    is2opt = false;
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
  void recut(int layer, pcl::PointCloud<PointType> &pl_feat_map, std::vector<Eigen::Vector3d> &pl_feat_eigen)//, pcl::PointCloud<PointXYZCOV> &pl_feat_map1)
  {
    if(plvec_tran->size() < MIN_PS)
    {
      feat_eigen_ratio = -1;
      if(layer != 0 && plvec_tran != nullptr) std::vector<Eigen::Vector3d>().swap(*plvec_tran);
      return;
    }

    PointType ap_centor_direct = calc_eigen(); // calculate eigenvalue ratio
    
    if(std::isnan(feat_eigen_ratio))
    {
      feat_eigen_ratio = -1;
      if(layer != 0 && plvec_tran != nullptr) std::vector<Eigen::Vector3d>().swap(*plvec_tran);
      return;
    }

    if(feat_eigen_ratio >= feat_eigen_limit)
    {
      pl_feat_map.push_back(ap_centor_direct);
      pl_feat_eigen.push_back(ap_enigen);
      // pl_feat_map1.push_back(ap_centor_cov);
      if(layer != 0 && plvec_tran != nullptr) std::vector<Eigen::Vector3d>().swap(*plvec_tran);
      return;
    }

    if(layer == 2) 
    {
      if(plvec_tran != nullptr) std::vector<Eigen::Vector3d>().swap(*plvec_tran);
      return;
    }

    int leafnum;

    for(uint j=0; j<plvec_tran->size(); j++)
    {
      int xyz[3] = {0, 0, 0};
      for(uint k=0; k<3; k++)
      {
        if((*plvec_tran)[j][k] > voxel_center[k])
        {
          xyz[k] = 1;
        }
      }
      leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
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
    for(uint i=0; i<8; i++)
    {
      if(leaves[i] != nullptr) leaves[i]->recut(layer, pl_feat_map, pl_feat_eigen);//, pl_feat_map1);
    }
  }
};