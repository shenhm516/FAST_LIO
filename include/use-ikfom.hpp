#ifndef USE_IKFOM_H
#define USE_IKFOM_H

#include <IKFoM_toolkit/esekfom/esekfom.hpp>

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos))
((SO3, rot))
((vect3, vel))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, omg))
((vect3, acc))
((vect3, pos_cur))
((SO3, rot_cur))
((vect3, vel_cur))
((vect3, bg))
((vect3, ba))
((S2, grav))
);

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
((vect3, acc_cur))
((vect3, gyro_cur))
);

MTK_BUILD_MANIFOLD(process_noise_ikfom,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{
	MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
	MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng, 0.0001);// 0.03
	MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na, 0.0001); // *dt 0.01 0.01 * dt * dt 0.05
	MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg, 0.00001); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba, 0.00001);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

//double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
//vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
Eigen::Matrix<double, 39, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	// std::cout << "Hello f 0" << std::endl;
	Eigen::Matrix<double, 39, 1> res = Eigen::Matrix<double, 39, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	vect3 omega_cur;
	in.gyro_cur.boxminus(omega_cur, s.bg);
	vect3 a_inertial = s.rot * (in.acc-s.ba); 
	vect3 a_inertial_cur = s.rot_cur * (in.acc_cur-s.ba); 
	for(int i = 0; i < 3; i++ ) {
		if (in.acc != Zero3d || in.gyro != Zero3d) {
			res(i) = s.vel[i];
			res(i + 3) = omega[i]; 
			res(i + 6) = a_inertial[i] + s.grav[i];
		}
		if (in.acc_cur != Zero3d || in.gyro_cur != Zero3d) {
			res(i + 21) = s.vel_cur[i];
			res(i + 24) = omega_cur[i]; 
			res(i + 27) = a_inertial_cur[i] + s.grav[i];
		} 
	}
	// std::cout << "Hello f 1" << std::endl;
	return res;
}

Eigen::Matrix<double, 39, 38> df_dx(state_ikfom &s, const input_ikfom &in)
{
	// std::cout << "Hello F 0" << std::endl;
	Eigen::Matrix<double, 39, 38> cov = Eigen::Matrix<double, 39, 38>::Zero();
	// vect3 omega;
	// in.gyro.boxminus(omega, s.bg);
	if (in.acc != Zero3d || in.gyro != Zero3d) {
		cov.template block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
		vect3 acc_;
		in.acc.boxminus(acc_, s.ba);
		cov.template block<3, 3>(6, 3) = -s.rot.toRotationMatrix()*MTK::hat(acc_);
		cov.template block<3, 3>(6, 33) = -s.rot.toRotationMatrix();
		Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
		Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
		s.S2_Mx(grav_matrix, vec, 36);
		cov.template block<3, 2>(6, 36) =  grav_matrix; 
		cov.template block<3, 3>(3, 30) = -Eigen::Matrix3d::Identity(); 
	}
	if (in.acc_cur != Zero3d || in.gyro_cur != Zero3d) {
		cov.template block<3, 3>(21, 27) = Eigen::Matrix3d::Identity(); 
		cov.template block<3, 3>(24, 30) = -Eigen::Matrix3d::Identity(); 
		vect3 acc_cur;
		in.acc_cur.boxminus(acc_cur, s.ba);
		cov.template block<3, 3>(27, 24) = -s.rot_cur.toRotationMatrix()*MTK::hat(acc_cur);
		cov.template block<3, 3>(27, 33) = -s.rot_cur.toRotationMatrix();
	}
	// std::cout << "Hello F 0" << std::endl;
	return cov;
}


Eigen::Matrix<double, 39, 12> df_dw(state_ikfom &s, const input_ikfom &in)
{
	// std::cout << "Hello Fw 0" << std::endl;
	Eigen::Matrix<double, 39, 12> cov = Eigen::Matrix<double, 39, 12>::Zero();
	if (in.acc != Zero3d || in.gyro != Zero3d) {
		cov.template block<3, 3>(6, 3) = -s.rot.toRotationMatrix();
		cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
		cov.template block<3, 3>(15, 0) = Eigen::Matrix3d::Identity();
		cov.template block<3, 3>(18, 3) = Eigen::Matrix3d::Identity();
		cov.template block<3, 3>(30, 6) = Eigen::Matrix3d::Identity();
		cov.template block<3, 3>(33, 9) = Eigen::Matrix3d::Identity();
	}
	if (in.acc_cur != Zero3d || in.gyro_cur != Zero3d) {
		cov.template block<3, 3>(27, 3) = -s.rot_cur.toRotationMatrix();
		cov.template block<3, 3>(24, 0) = -Eigen::Matrix3d::Identity();
	}
	// std::cout << "Hello Fw 1" << std::endl;
	return cov;
}

vect3 SO3ToEuler(const SO3 &orient) 
{
	Eigen::Matrix<double, 3, 1> _ang;
	Eigen::Vector4d q_data = orient.coeffs().transpose();
	//scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1], z=orient.coeffs[2];
	double sqw = q_data[3]*q_data[3];
	double sqx = q_data[0]*q_data[0];
	double sqy = q_data[1]*q_data[1];
	double sqz = q_data[2]*q_data[2];
	double unit = sqx + sqy + sqz + sqw; // if normalized is one, otherwise is correction factor
	double test = q_data[3]*q_data[1] - q_data[2]*q_data[0];

	if (test > 0.49999*unit) { // singularity at north pole
	
		_ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
	if (test < -0.49999*unit) { // singularity at south pole
		_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
		
	_ang <<
			std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw),
			std::asin (2*test/unit),
			std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);
	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	vect3 euler_ang(temp, 3);
		// euler_ang[0] = roll, euler_ang[1] = pitch, euler_ang[2] = yaw
	return euler_ang;
}

#endif