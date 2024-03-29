#pragma once

#include <Eigen/Dense>
#include "generated/forward_dynamics.h"
#include "generated/model_constants.h"

using Jackal::rcg::Scalar;

void roty(Eigen::Matrix<Scalar,3,3>& rot, Scalar angle);

//This function uses a first order approximation of quaternion_dot. Works well with RK4 and low timestep.
Eigen::Matrix<Scalar,4,1> calcQuatDot(Eigen::Matrix<Scalar,4,1> orientation, Eigen::Matrix<Scalar,3,1> ang_vel_body);

//Say quaternion represents the orientation of Frame B wtr to Frame A
//Then this returns a matrix representation of the rotational
//displacement from A to B.
//The rotation that converts a frame coincident with A to frame B
Eigen::Matrix<Scalar,3,3> toMatrixRotation(Eigen::Matrix<Scalar,4,1> quaternion);
Eigen::Matrix<Scalar,3,3> toMatrixRotation(Scalar x, Scalar y, Scalar z, Scalar w);

void toEulerAngles(const Scalar &qw,
		   const Scalar &qx,
		   const Scalar &qy,
		   const Scalar &qz,
		   Scalar &roll,
		   Scalar &pitch,
		   Scalar &yaw);
