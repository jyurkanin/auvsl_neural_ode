#include "BekkerDynamics.h"

BekkerDynamics::BekkerDynamics()
{

}
BekkerDynamics::~BekkerDynamics()
{
	
}

void BekkerDynamics::setParams(const VectorS &params)
{
	for(int i = 0; i < m_params.size(); i++)
	{
		m_params[i] = params[i];
	}
}

void BekkerDynamics::getParams(VectorS &params)
{
	for(int i = 0; i < m_params.size(); i++)
	{
		params[i] = m_params[i];
	}
}

void BekkerDynamics::get_tire_f_ext(const Eigen::Matrix<Scalar,STATE_DIM,1> &X, LinkDataMap<Force> &ext_forces)
{
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
	Eigen::Matrix<Scalar,4,1> forces;
	features[3] = m_params[0];
	features[4] = m_params[1];
	features[5] = m_params[2];
	features[6] = m_params[3];
	features[7] = m_params[4];
  
	for(int ii = 0; ii < 4; ii++){    
		//17 is the idx that tire velocities start at.
		Scalar vel_x_tan = .098*X[17+ii];
		Scalar tire_vx = cpt_vels[ii][0];

		Scalar small = 1e-4;
		vel_x_tan = CppAD::CondExpGt(vel_x_tan, small, vel_x_tan, small);
		tire_vx = CppAD::CondExpGt(tire_vx, small, tire_vx, small);
		
		features[0] = sinkages[ii];
		features[1] = (vel_x_tan - tire_vx)/vel_x_tan; // This will need a sign correction on the output
		features[2] = CppAD::tanh(cpt_vels[ii][1] / tire_vx);
		
		if(sinkages[ii] > 0.0)
		{		
			forces = m_bekker_tire_model.get_forces(features);
		}
		else
		{
			forces[0] = 0.0;
			forces[1] = 0.0;
			forces[2] = 0.0;
			forces[3] = 0.0;
		}
		
		forces[0] = CppAD::CondExpGt((vel_x_tan - cpt_vels[ii][0]), small, forces[0], -forces[0]);
		
		Eigen::Matrix<Scalar,3,1> lin_force;    
-		lin_force[0] = forces[0];
		lin_force[1] = forces[1];
		lin_force[2] = forces[2];
	
		// Convert from world orientation to tire_cpt orientation
		// So that reaction forces are oriented with the surface normal
		Eigen::Matrix<Scalar,3,1> temp_vel = cpt_rots[ii]*cpt_vels[ii];
    
		// ang_force = cpt_rots[ii].transpose()*ang_force;
		// Numerical hack to help stabilize sinkage
		lin_force[2] += -200.0*cpt_vels[ii][2]; //dead simple, works fine.
		
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
