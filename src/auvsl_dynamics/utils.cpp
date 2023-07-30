#include "utils.h"



void roty(Eigen::Matrix<Scalar,3,3>& rot, Scalar angle){
	rot(0,0) = CppAD::cos(angle);
	rot(0,1) = 0;
	rot(0,2) = CppAD::sin(angle);
	
	rot(1,0) = 0;
	rot(1,1) = 1;
	rot(1,2) = 0;
	
	rot(2,0) = -CppAD::sin(angle);
	rot(2,1) = 0;
	rot(2,2) = CppAD::cos(angle);
}


Eigen::Matrix<Scalar,4,1> calcQuatDot(Eigen::Matrix<Scalar,4,1> orientation, Eigen::Matrix<Scalar,3,1> ang_vel_body){
  Eigen::Matrix<Scalar,4,3> m;
  m(0, 0) =  orientation[3];   m(0, 1) = -orientation[2];   m(0, 2) =  orientation[1];
  m(1, 0) =  orientation[2];   m(1, 1) =  orientation[3];   m(1, 2) = -orientation[0];
  m(2, 0) = -orientation[1];   m(2, 1) =  orientation[0];   m(2, 2) =  orientation[3];
  m(3, 0) = -orientation[0];   m(3, 1) = -orientation[1];   m(3, 2) = -orientation[2];
  
  return .5 * m * ang_vel_body;
}



//https://github.com/rbdl/rbdl/blob/master/include/rbdl/Quaternion.h
//Say quaternion represents the orientation of Frame B wtr to Frame A
//Then this returns a matrix representation of the rotational
//displacement from A to B.
//The rotation that converts a frame coincident with A to frame B
Eigen::Matrix<Scalar,3,3> toMatrixRotation(Eigen::Matrix<Scalar,4,1> quaternion) {
  Scalar x = quaternion[0];
  Scalar y = quaternion[1];
  Scalar z = quaternion[2];
  Scalar w = quaternion[3];
  return toMatrixRotation(x,y,z,w);
}

Eigen::Matrix<Scalar,3,3> toMatrixRotation(Scalar x, Scalar y, Scalar z, Scalar w) {
  Eigen::Matrix<Scalar,3,3> rot;
  rot <<
    1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
    2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y;
  return rot;
}

// whos gonna stop me
// copy and pasted from youtube
void toEulerAngles(const Scalar &qw,
		   const Scalar &qx,
		   const Scalar &qy,
		   const Scalar &qz,
		   Scalar &roll,
		   Scalar &pitch,
		   Scalar &yaw)
{
    // roll (x-axis rotation)
    Scalar sinr_cosp = 2 * (qw * qx + qy * qz);
    Scalar cosr_cosp = 1 - 2 * (qx * qx + qy * qy);
    roll = CppAD::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    Scalar sinp = CppAD::sqrt(1 + 2 * (qw * qy - qx * qz));
    Scalar cosp = CppAD::sqrt(1 - 2 * (qw * qy - qx * qz));
    pitch = 2 * CppAD::atan2(sinp, cosp) - M_PI / 2; // Why do we subtract pi/2?

    // yaw (z-axis rotation)
    Scalar siny_cosp = 2 * (qw * qz + qx * qy);
    Scalar cosy_cosp = 1 - 2 * (qy * qy + qz * qz);
    yaw = CppAD::atan2(siny_cosp, cosy_cosp);
}
