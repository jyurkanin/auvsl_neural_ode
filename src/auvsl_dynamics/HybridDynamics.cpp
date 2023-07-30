#include "HybridDynamics.h"
#include "generated/declarations.h"
#include "generated/miscellaneous.h"
#include "utils.h"
#include <iit/rbd/robcogen_commons.h>
#include <cppad/cppad.hpp>
#include <cppad/utility/to_string.hpp>
#include <iostream>

using CppAD::AD;


using Jackal::rcg::tire_radius;
using Jackal::rcg::tx_front_left_wheel;
using Jackal::rcg::ty_front_left_wheel;
using Jackal::rcg::tz_front_left_wheel;

using Jackal::rcg::tx_front_right_wheel;
using Jackal::rcg::ty_front_right_wheel;
using Jackal::rcg::tz_front_right_wheel;

using Jackal::rcg::tx_rear_left_wheel;
using Jackal::rcg::ty_rear_left_wheel;
using Jackal::rcg::tz_rear_left_wheel;

using Jackal::rcg::tx_rear_right_wheel;
using Jackal::rcg::ty_rear_right_wheel;
using Jackal::rcg::tz_rear_right_wheel;

using Jackal::rcg::orderedJointIDs;


const Scalar HybridDynamics::timestep = 1e-3;
const Acceleration HybridDynamics::GRAVITY_VEC = (Acceleration() << 0,0,0,0,0,-9.81).finished();

//std::ofstream HybridDynamics::log_file;
//std::ofstream HybridDynamics::debug_file;

Scalar get_altitude(const Scalar &x, const Scalar &y)
{
	return Scalar(0.0);
}

HybridDynamics::HybridDynamics(){
  fwd_dynamics = new Jackal::rcg::ForwardDynamics(inertias, m_transforms);
  
  //For more information on these, see TerrainMap.cpp and TerrainMap.h
  
  bekker_params[0] = 29.758547;
  bekker_params[1] = 2083.000000;
  bekker_params[2] = 1.197933;
  bekker_params[3] = 0.102483;
  bekker_params[4] = 0.652405;
  
  JointState q(0,0,0,0);   //Joint position
  f_transforms.fr_front_left_wheel_link_X_fr_base_link.update(q);
  f_transforms.fr_base_link_X_fr_front_left_wheel_link_COM.update(q);
}

HybridDynamics::~HybridDynamics(){
  delete fwd_dynamics;
}

void HybridDynamics::initState(){
  Scalar start_state[STATE_DIM] = {0,0,0,1, 0,0,.16, 0,0,0,0, 0,0,0, 0,0,0, 0,0,0,0};
  initState(start_state);
}

void HybridDynamics::initState(Scalar *start_state){
  for(unsigned i = 0; i < STATE_DIM; i++){
    state_[i] = start_state[i];
  }
}

void HybridDynamics::convertBaseToCOM(VectorS &com_state, const VectorS &base_state)
{
  Jackal::rcg::Vector3 com_pos = Jackal::rcg::getWholeBodyCOM(inertias, h_transforms);
  
  Eigen::Matrix<Scalar,3,1> com_lin_vel;
  Eigen::Matrix<Scalar,3,1> base_lin_vel(base_state[14], base_state[15], base_state[16]);
  Eigen::Matrix<Scalar,3,1> ang_vel(base_state[11], base_state[12], base_state[13]);
  
  Eigen::Matrix<Scalar,3,3> r_ss;
  r_ss << 0.,        -com_pos[2], com_pos[1],
          com_pos[2], 0.,        -com_pos[0],
         -com_pos[1], com_pos[0], 0.;
  
  // minus, because com_pos is the displacement from com to base_link
  com_lin_vel = base_lin_vel - (r_ss*ang_vel);
  
  for(int i = 0; i < STATE_DIM; i++){
    com_state[i] = base_state[i];
  }

  com_state[14] = com_lin_vel[0];
  com_state[15] = com_lin_vel[1];
  com_state[16] = com_lin_vel[2];
}

void HybridDynamics::convertCOMToBase(const VectorS &com_state, VectorS &base_state)
{
	Scalar com_state_arr[STATE_DIM];
	Scalar base_state_arr[STATE_DIM];

	for(int i = 0; i < STATE_DIM; i++)
	{
		com_state_arr[i] = com_state[i];
	}
	initStateCOM(com_state_arr, base_state_arr);
}

//velocities expressed in COM frame. more intuitive.
void HybridDynamics::initStateCOM(Scalar *start_state, Scalar *base_state){
  Jackal::rcg::Vector3 com_pos = Jackal::rcg::getWholeBodyCOM(inertias, h_transforms);
  
  Eigen::Matrix<Scalar,3,1> base_lin_vel;
  Eigen::Matrix<Scalar,3,1> com_lin_vel(start_state[14], start_state[15], start_state[16]);
  Eigen::Matrix<Scalar,3,1> ang_vel(start_state[11], start_state[12], start_state[13]);
  
  Eigen::Matrix<Scalar,3,3> r_ss;
  r_ss << 0.,        -com_pos[2], com_pos[1],
          com_pos[2], 0.,        -com_pos[0],
         -com_pos[1], com_pos[0], 0.;
  
  //plus, because com_pos is the displacement from base_link to com
  base_lin_vel = com_lin_vel + (r_ss*ang_vel);
  
  for(int i = 0; i < STATE_DIM; i++){
    base_state[i] = start_state[i];
  }

  base_state[14] = base_lin_vel[0];
  base_state[15] = base_lin_vel[1];
  base_state[16] = base_lin_vel[2];
}

void HybridDynamics::initStateCOM(Scalar *start_state)
{
  Scalar base_state[STATE_DIM];
  initStateCOM(start_state, base_state);
  initState(base_state);
}

void HybridDynamics::settle(){
  //reach equillibrium sinkage.
  for(int i = 0; i < 20; i++){
    step(0,0);
  }
}

//step is .05s
//THis matches up with the test dataset sample rate.
void HybridDynamics::step(Scalar vl, Scalar vr){  
  Eigen::Matrix<Scalar,STATE_DIM,1> Xt1;
  Eigen::Matrix<Scalar,CNTRL_DIM,1> u;
  
  u(0) = vl;
  u(1) = vr;
  
  const int num_steps = 50; //10*.005 = .05
  for(int ii = 0; ii < num_steps; ii++){
    state_[17] = state_[19] = u[0];
    state_[18] = state_[20] = u[1];
    
    RK4(state_, Xt1, u);
    //Euler(state_, Xt1, u);
    state_ = Xt1;
  }
}


//Going to assume tire contacts soil at  -tire_radius directly underneath the tire joint.
void HybridDynamics::get_tire_cpts(const Eigen::Matrix<Scalar,STATE_DIM,1> &X,
                                   Eigen::Matrix<Scalar,3,1> *cpt_pts,
                                   Eigen::Matrix<Scalar,3,3> *cpt_rots){
  const Eigen::Matrix<Scalar,3,4> tire_translations = (Eigen::Matrix<Scalar,3,4>() <<
                                                       tx_front_left_wheel,tx_front_right_wheel,tx_rear_left_wheel,tx_rear_right_wheel,
                                                       ty_front_left_wheel,ty_front_right_wheel,ty_rear_left_wheel,ty_rear_right_wheel,
                                                       tz_front_left_wheel,tz_front_right_wheel,tz_rear_left_wheel,tz_rear_right_wheel).finished();
  
  const Eigen::Matrix<Scalar,3,4> radius_matrix = (Eigen::Matrix<Scalar,3,4>() <<
                                                   0,0,0,0,
                                                   0,0,0,0,
                                                   -tire_radius,-tire_radius,-tire_radius,-tire_radius).finished();
  
  // get the matrix that transforms from base to world frame
  // This matrix also expresses 
  Eigen::Matrix<Scalar,3,3> rot = toMatrixRotation(X[0],X[1],X[2],X[3]);
  
  // this is like the end effector if it was a quadruped. 
  const Eigen::Matrix<Scalar,3,4> end_pos_matrix = rot*(tire_translations + radius_matrix);
  
  for(int ii = 0; ii < 4; ii++){
    cpt_rots[ii] = rot;
    cpt_pts[ii][0] = end_pos_matrix(0,ii) + X[4];
    cpt_pts[ii][1] = end_pos_matrix(1,ii) + X[5];
    cpt_pts[ii][2] = end_pos_matrix(2,ii) + X[6];
  }
}

void HybridDynamics::get_tire_sinkages(const Eigen::Matrix<Scalar,3,1> *cpt_points, Scalar *sinkages){
  const Scalar altitude = 0;
  sinkages[0] = altitude - cpt_points[0][2];
  sinkages[1] = altitude - cpt_points[1][2];
  sinkages[2] = altitude - cpt_points[2][2];
  sinkages[3] = altitude - cpt_points[3][2];
}


// Actually iterate and find where the tire is sinking into the soil the most
// And call that the contact point. Then report the orientation of this cpt.
void HybridDynamics::get_tire_cpts_sinkages_fancy(const Eigen::Matrix<Scalar,STATE_DIM,1> &X,
												  Eigen::Matrix<Scalar,3,1> *cpt_pts,
												  Eigen::Matrix<Scalar,3,3> *cpt_rots,
												  Scalar *sinkages){

	const Eigen::Matrix<Scalar,3,4> tire_translations = (Eigen::Matrix<Scalar,3,4>() <<
														 tx_front_left_wheel,tx_front_right_wheel,tx_rear_left_wheel,tx_rear_right_wheel,
														 ty_front_left_wheel,ty_front_right_wheel,ty_rear_left_wheel,ty_rear_right_wheel,
														 tz_front_left_wheel,tz_front_right_wheel,tz_rear_left_wheel,tz_rear_right_wheel).finished();

	const Eigen::Matrix<Scalar,3,1> radius_vec = (Eigen::Matrix<Scalar,3,1>() << 0,0,-tire_radius).finished();

	//get the matrix that transforms from base to world frame
	Eigen::Matrix<Scalar,3,3> base_rot = toMatrixRotation(X[0],X[1],X[2],X[3]);

	Eigen::Matrix<Scalar,3,4> end_pos_matrix = base_rot*tire_translations;
	Eigen::Matrix<Scalar,3,1> end_pos_tire_joint;

	int max_checks = 10;
	Scalar test_angle;
	Scalar max_angle = -.3; //arbitrary
	Scalar test_sinkage;
	Scalar max_sinkage;

	Eigen::Matrix<Scalar,3,1> test_pos;
	Eigen::Matrix<Scalar,3,3> test_rot;
	Eigen::Matrix<Scalar,3,3> best_rot;

	for(int ii = 0; ii < 4; ii++){
		end_pos_tire_joint[0] = end_pos_matrix(0,ii) + X[4]; //real world position of tire joint.
		end_pos_tire_joint[1] = end_pos_matrix(1,ii) + X[5];
		end_pos_tire_joint[2] = end_pos_matrix(2,ii) + X[6];

		cpt_pts[ii] = end_pos_tire_joint; //center of the tire joint.

		//need to find the cpt point between tire and soil to determine the
		//orientation of the reaction forces frame.

		max_sinkage = 0;
		best_rot = Eigen::Matrix<Scalar,3,3>::Identity();
		for(int jj = 0; jj < max_checks; jj++){
			test_angle = max_angle - (2*max_angle*jj/((Scalar) max_checks - 1)); //test_angle ranges from +-max_angle
			roty(test_rot, test_angle);
			test_rot = base_rot*test_rot;

			test_pos = end_pos_tire_joint + test_rot*radius_vec; //a point on the edge of the tire.
			test_sinkage = get_altitude(test_pos[0],test_pos[1]) - test_pos[2];

			if(test_sinkage > max_sinkage){
				best_rot = test_rot; //orientation of cpt
				max_sinkage = test_sinkage;
			}
		}

		sinkages[ii] = max_sinkage;
		cpt_rots[ii] = best_rot;
	}
}




void HybridDynamics::get_tire_cpt_vels(const Eigen::Matrix<Scalar,STATE_DIM,1> &X, Eigen::Matrix<Scalar,3,1> *cpt_vels){
  const Eigen::Matrix<Scalar,3,4> tire_translations = (Eigen::Matrix<Scalar,3,4>() <<
                                                       tx_front_left_wheel,tx_front_right_wheel,tx_rear_left_wheel,tx_rear_right_wheel,
                                                       ty_front_left_wheel,ty_front_right_wheel,ty_rear_left_wheel,ty_rear_right_wheel,
                                                       tz_front_left_wheel,tz_front_right_wheel,tz_rear_left_wheel,tz_rear_right_wheel).finished();
  
  const Eigen::Matrix<Scalar,3,4> radius_matrix = (Eigen::Matrix<Scalar,3,4>() <<
                                                   0,0,0,0,
                                                   0,0,0,0,
                                                   -tire_radius,-tire_radius,-tire_radius,-tire_radius).finished();
  
  // This is like the end effector if it was a quadruped. 
  const Eigen::Matrix<Scalar,3,4> end_pos_matrix = tire_translations + radius_matrix;
  
  // vel of base_link expressed in base link frame.
  Eigen::Matrix<Scalar,3,1> lin_vel(X[14], X[15], X[16]);
  Eigen::Matrix<Scalar,3,1> ang_vel(X[11], X[12], X[13]);

  for(int ii = 0; ii < 4; ii++){
    Eigen::Matrix<Scalar,3,3> r_ss;
    r_ss << 0., -end_pos_matrix(2,ii), end_pos_matrix(1,ii),
      end_pos_matrix(2,ii), 0., -end_pos_matrix(0,ii),
      -end_pos_matrix(1,ii), end_pos_matrix(0,ii), 0.;
    
    cpt_vels[ii] = lin_vel - (r_ss*ang_vel);
  }
  
  // Only interested in the linear velocities of the cpt points. Maybe we can include angular later.
}

void HybridDynamics::get_tire_f_ext(const Eigen::Matrix<Scalar,STATE_DIM,1> &X, LinkDataMap<Force> &ext_forces){
	Eigen::Matrix<Scalar,3,1> cpt_points[4];  //world frame position of cpt frames
	Eigen::Matrix<Scalar,3,3> cpt_rots[4];    //world orientation of cpt frames
	get_tire_cpts(X, cpt_points, cpt_rots);
  
	Scalar sinkages[4];
    get_tire_sinkages(cpt_points, sinkages);
  
	//get the velocity of each tire contact point expressed in the contact point frame
	Eigen::Matrix<Scalar,3,1> cpt_vels[4];
	get_tire_cpt_vels(X, cpt_vels); 
	
	// We now have sinkage and velocity of each tire contact point
	// Next we need to compute tire-soil reaction forces
	// Then we will transform these forces into the body frame of each tire
	// Due to the stupid way that the tire joint frame transforms are defined we
	// will need a transform that undos the rotation of the tire.
	// A smarter solution would be to permanently set the tire joint angles to zero
	// because those values literally change nothing about the simulation.
	// We're doing it: Joint positions are set to zero.
	// Transform is needed to tire frame because joints are oriented so that z is the joint axis.
  
	Eigen::Matrix<Scalar,8,1> features;
	Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> forces;
	features[4] = 0;
	features[5] = 0;
	features[6] = 0;
	features[7] = 0;
  
	for(int ii = 0; ii < 4; ii++){    
		//17 is the idx that tire velocities start at.    
		features[0] = cpt_vels[ii][0];
		features[1] = cpt_vels[ii][1];
		features[2] = X[17+ii];
		features[3] = sinkages[ii];

		tire_network.forward(features, forces, 0);
		
		m_penalty += forces[3];
		
		Eigen::Matrix<Scalar,3,1> lin_force;    
		lin_force[0] = forces[0];
		lin_force[1] = forces[1];
		lin_force[2] = forces[2];
	
		// Convert from world orientation to tire_cpt orientation
		// So that reaction forces are oriented with the surface normal
		Eigen::Matrix<Scalar,3,1> temp_vel = cpt_rots[ii]*cpt_vels[ii];
    
		// ang_force = cpt_rots[ii].transpose()*ang_force;
		// Numerical hack to help stabilize sinkage
		// todo: replace cpt_vels with temp_vel
		lin_force[2] += -200*cpt_vels[ii][2]; //dead simple, works fine.
		
		Force wrench;
		wrench[0] = 0;
		wrench[1] = 0;
		wrench[2] = 0;
		wrench[3] = lin_force[0];  // the different indices here is due to a rotation in coordinate frame.
		wrench[4] = -lin_force[2]; // normal force Fz maps to force in Y direction due to Robcogen's choice of coordinate frame for joints.
		wrench[5] = lin_force[1];

		// ext_forces are expressed in the frame of the link
		// Double check this. Yeah it says so in forward_dynamics.h
		ext_forces[orderedLinkIDs[ii+1]] = wrench;  
	}
}


// Don't use. Didn't help.
void HybridDynamics::get_base_f_ext(const Eigen::Matrix<Scalar,STATE_DIM,1> &X, LinkDataMap<Force> &ext_forces)
{
	// get the matrix that transforms from base to world frame
	// This matrix also expresses 
	Eigen::Matrix<Scalar,3,3> rot = toMatrixRotation(X[0],X[1],X[2],X[3]);
	
	Eigen::Matrix<Scalar,3,1> features;
	Eigen::Matrix<Scalar,3,1> forces;

	features[0] = X[13]; //wz
	features[1] = X[14]; //vx
	features[2] = X[15]; //vy
	//base_network.forward(features, forces);
	
	Eigen::Matrix<Scalar,3,1> ang_force;
	ang_force[0] = 0;
	ang_force[1] = 0;
	ang_force[2] = forces[0]; //nz	
	
	Eigen::Matrix<Scalar,3,1> lin_force;
	lin_force[0] = forces[1]; //fx
	lin_force[1] = forces[2]; //fy
	lin_force[2] = 0;
	
	Force wrench;
	wrench[0] =  ang_force[0]; // 0
	wrench[1] = -ang_force[2]; // nz
	wrench[2] =  ang_force[1]; // 0
	
	wrench[3] =  lin_force[0]; // Fx
	wrench[4] = -lin_force[2]; // Actually ignored
	wrench[5] =  lin_force[1]; // Fy
	
	// 0 is the index corresponding to the base body
	ext_forces[orderedLinkIDs[0]] = wrench;
	
}



void HybridDynamics::RK4(const Eigen::Matrix<Scalar,STATE_DIM,1> &X, Eigen::Matrix<Scalar,STATE_DIM,1> &Xt1, Eigen::Matrix<Scalar,CNTRL_DIM,1> &u){
  Eigen::Matrix<Scalar,STATE_DIM,1> temp;
  Eigen::Matrix<Scalar,STATE_DIM,1> k1;
  Eigen::Matrix<Scalar,STATE_DIM,1> k2;
  Eigen::Matrix<Scalar,STATE_DIM,1> k3;
  Eigen::Matrix<Scalar,STATE_DIM,1> k4;
  Scalar temp_norm;
  Scalar ts = timestep;
  
  ODE(X, k1, u);
  temp = X + .5*ts*k1;
  temp_norm = CppAD::sqrt(temp[0]*temp[0] + temp[1]*temp[1] + temp[2]*temp[2] + temp[3]*temp[3]);
  temp[0] /= temp_norm;
  temp[1] /= temp_norm;
  temp[2] /= temp_norm;
  temp[3] /= temp_norm;
  
  ODE(temp, k2, u);
  temp = X + .5*ts*k2;
  temp_norm = CppAD::sqrt(temp[0]*temp[0] + temp[1]*temp[1] + temp[2]*temp[2] + temp[3]*temp[3]);
  temp[0] /= temp_norm;
  temp[1] /= temp_norm;
  temp[2] /= temp_norm;
  temp[3] /= temp_norm;
  
  ODE(temp, k3, u);
  temp = X + ts*k3;
  temp_norm = CppAD::sqrt(temp[0]*temp[0] + temp[1]*temp[1] + temp[2]*temp[2] + temp[3]*temp[3]);
  temp[0] /= temp_norm;
  temp[1] /= temp_norm;
  temp[2] /= temp_norm;
  temp[3] /= temp_norm;
  
  ODE(temp, k4, u);
  Xt1 = X + (ts/6.0)*(k1 + 2*k2 + 2*k3 + k4);
  temp_norm = CppAD::sqrt(Xt1[0]*Xt1[0] + Xt1[1]*Xt1[1] + Xt1[2]*Xt1[2] + Xt1[3]*Xt1[3]);
  Xt1[0] /= temp_norm;
  Xt1[1] /= temp_norm;
  Xt1[2] /= temp_norm;
  Xt1[3] /= temp_norm;

}

void HybridDynamics::Euler(const Eigen::Matrix<Scalar,STATE_DIM,1> &X, Eigen::Matrix<Scalar,STATE_DIM,1> &Xt1, Eigen::Matrix<Scalar,CNTRL_DIM,1> &u){
  Scalar ts = timestep;
  Eigen::Matrix<Scalar,STATE_DIM,1> Xd;
  ODE(X, Xd, u);
  Xt1 = X + (Xd*ts);
  
  Scalar temp_norm = CppAD::sqrt(Xt1[0]*Xt1[0] + Xt1[1]*Xt1[1] + Xt1[2]*Xt1[2] + Xt1[3]*Xt1[3]);
  Xt1[0] /= temp_norm;
  Xt1[1] /= temp_norm;
  Xt1[2] /= temp_norm;
  Xt1[3] /= temp_norm;
}

//Vehicle ODE
//A combination of forward dynamics, external forces, and quaternion derivative.
void HybridDynamics::ODE(const Eigen::Matrix<Scalar,STATE_DIM,1> &X, Eigen::Matrix<Scalar,STATE_DIM,1> &Xd, Eigen::Matrix<Scalar,CNTRL_DIM,1> &u){
  const JointState q(0,0,0,0);   //Joint position
  JointState qd;  //velocity
  JointState qdd; //acceleration
  const JointState tau(0,0,0,0); //Joint torque
  //Not going to use this ^^^^, going to just set tire velocity directly.
  //Assumption is that tire motor controllers are really good and can reach whatever velocity we command.
  //Later on we can use neural networks to estimate joint torque.
  
  Acceleration base_acc; //base refers to the floating base. Not world frame. 
  Velocity base_vel;
  
  //Could experiment with setting these to 0, as they dont change inertia properties.
  //And setting them to a constant 0 could enable optimizations and cancel stuff. Idk.
  //I'm doing it.
  // q[0] = q1 //tire positions.
  // q[1] = q2
  // q[2] = q3
  // q[3] = q4
    
  qd[0] = u[0]; //q1 //tire velocities. These actually matter.
  qd[1] = u[1]; //q2
  qd[2] = u[0]; //q3
  qd[3] = u[1]; //q4
  
  base_vel[0] = X[11];
  base_vel[1] = X[12];
  base_vel[2] = X[13];
  base_vel[3] = X[14];
  base_vel[4] = X[15];
  base_vel[5] = X[16];
  
  //Each force is expressed in the respective link frame.
  //intializes all forces to 0.
  LinkDataMap<Force> ext_forces(Force::Zero());
  
  //Calculates ext forces
  get_tire_f_ext(X, ext_forces);
  // get_base_f_ext(X, ext_forces);
  
  //rot is the rotation matrix that transforms vectors from the base frame to world frame.
  //The transpose converts the world frame gravity to a base frame gravity vector.
  Eigen::Matrix<Scalar,4,1> quat(X[0], X[1], X[2], X[3]);
  Eigen::Matrix<Scalar,3,3> rot = toMatrixRotation(X[0],X[1],X[2],X[3]);
  Eigen::Matrix<Scalar,3,1> gravity_lin(GRAVITY_VEC[3], GRAVITY_VEC[4], GRAVITY_VEC[5]);
  gravity_lin = rot.transpose()*gravity_lin;
  
  Force gravity_b;
  gravity_b[0] = 0;
  gravity_b[1] = 0;
  gravity_b[2] = 0;
  gravity_b[3] = gravity_lin[0];
  gravity_b[4] = gravity_lin[1];
  gravity_b[5] = gravity_lin[2];
  
  //this calculates base_acc and qdd.
  fwd_dynamics->fd(qdd, base_acc, base_vel, gravity_b, q, qd, tau, ext_forces);  
  
  // Velocity com_vel = m_transforms.fr_base_link_COM_X_fr_base_link*base_vel;
  // Acceleration com_acc = m_transforms.fr_base_link_COM_X_fr_base_link*base_acc;

  // std::cout << "qdd[0]: " << qdd[0] << "\n";
  // std::cout << "com_vel: "
  // 			<< CppAD::Value(com_vel[0]) << ", "
  // 			<< CppAD::Value(com_vel[1]) << ", "
  // 			<< CppAD::Value(com_vel[2]) << ", "
  // 			<< CppAD::Value(com_vel[3]) << ", "
  // 			<< CppAD::Value(com_vel[4]) << ", "
  // 			<< CppAD::Value(com_vel[5]) << "\n";
  // std::cout << "com_acc: "
  // 			<< CppAD::Value(com_acc[0]) << ", "
  // 			<< CppAD::Value(com_acc[1]) << ", "
  // 			<< CppAD::Value(com_acc[2]) << ", "
  // 			<< CppAD::Value(com_acc[3]) << ", "
  // 			<< CppAD::Value(com_acc[4]) << ", "
  // 			<< CppAD::Value(com_acc[5]) << "\n";
  
  // #include <cstdlib>
  // exit(1);
  
  Eigen::Matrix<Scalar,3,1> ang_vel(base_vel[0], base_vel[1], base_vel[2]);
  Eigen::Matrix<Scalar,3,1> lin_vel(base_vel[3], base_vel[4], base_vel[5]);
  Eigen::Matrix<Scalar,4,1> quat_dot = calcQuatDot(quat, ang_vel);
  
  lin_vel = rot*lin_vel; // converts from base to world frame.
  
  Xd[0] = quat_dot[0];
  Xd[1] = quat_dot[1];
  Xd[2] = quat_dot[2];
  Xd[3] = quat_dot[3];
  
  Xd[4] = lin_vel[0];
  Xd[5] = lin_vel[1];
  Xd[6] = lin_vel[2];
  
  Xd[7]  = u[0]; //X[17];
  Xd[8]  = u[1]; //X[18];
  Xd[9]  = u[0]; //X[19];
  Xd[10] = u[1]; //X[20];
  
  Xd[11] = base_acc[0];
  Xd[12] = base_acc[1];
  Xd[13] = base_acc[2];
  Xd[14] = base_acc[3];
  Xd[15] = base_acc[4];
  Xd[16] = base_acc[5];
  
  Xd[17] = 0; //qdd[0];
  Xd[18] = 0; //qdd[1];
  Xd[19] = 0; //qdd[2];
  Xd[20] = 0; //qdd[3];
}
