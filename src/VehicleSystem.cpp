#include "VehicleSystem.h"
#include "utils.h"

#include <iostream>
#include <assert.h>

template<typename Scalar>
VehicleSystem<Scalar>::VehicleSystem()
{
	this->setNumParams(m_hybrid_dynamics.tire_network.getNumParams());
	this->setStateDim(HybridDynamics::STATE_DIM);
	this->setControlDim(HybridDynamics::CNTRL_DIM);
}

template<typename Scalar>
VehicleSystem<Scalar>::~VehicleSystem()
{
  
}

template<typename Scalar>
void VehicleSystem<Scalar>::setParams(const VectorS &params)
{
	int idx = 0;
	m_hybrid_dynamics.tire_network.setParams(params, idx);
}

template<typename Scalar>
void VehicleSystem<Scalar>::getParams(VectorS &params)
{
	int idx = 0;
	m_hybrid_dynamics.tire_network.getParams(params, idx);
}

template<typename Scalar>
void VehicleSystem<Scalar>::forward(const VectorS &X, VectorS &Xd)
{
	assert(X.size() == this->getStateDim());
	assert(Xd.size() == this->getStateDim());
  
	Eigen::Matrix<Scalar, HybridDynamics::STATE_DIM, 1> model_x;
	Eigen::Matrix<Scalar, HybridDynamics::CNTRL_DIM, 1> model_u;
	Eigen::Matrix<Scalar, HybridDynamics::STATE_DIM, 1> model_xd;
  
	for(int i = 0; i < model_x.size(); i++)
	{
		model_x[i] = X[i];
	}
	for(int i = 0; i < model_u.size(); i++)
	{
		model_u[i] = X[i+model_x.size()];
	}
  
	m_hybrid_dynamics.ODE(model_x, model_xd, model_u);
  
	for(int i = 0; i < model_xd.size(); i++)
	{
		Xd[i] = model_xd[i];
	}
	for(int i = 0; i < model_u.size(); i++)
	{
		Xd[i+model_x.size()] = 0;
	}  
}

// Linear and Angular error
template<typename Scalar>
Scalar VehicleSystem<Scalar>::loss(const VectorS &gt_vec, VectorS &vec)
{
	// Linear Error
	Scalar x_err = gt_vec[4] - vec[4];
	Scalar y_err = gt_vec[5] - vec[5];
	Scalar lin_err = CppAD::sqrt((x_err*x_err) + (y_err*y_err));

	// Angular Error
	Scalar roll, pitch, yaw;
	toEulerAngles(vec[3], vec[0], vec[1], vec[2],
				  roll, pitch, yaw);
	
	Scalar yaw_err = yaw - gt_vec[3];
	yaw_err = CppAD::atan2(CppAD::sin(yaw_err), CppAD::cos(yaw_err));
	Scalar ang_err = CppAD::abs(yaw_err);
	
	Scalar wz_err = CppAD::abs(gt_vec[13] - vec[13]);
	Scalar vx_err = CppAD::abs(gt_vec[14] - vec[14]);
	Scalar vy_err = CppAD::abs(gt_vec[15] - vec[15]);
	Scalar vel_err = .1*(wz_err+vx_err+vy_err);

	return ang_err + lin_err;
}


template<typename Scalar>
void VehicleSystem<Scalar>::evaluate(const VectorS &gt_vec, const VectorS &vec, Scalar &ang_mse, Scalar &lin_mse)
{
  Scalar x_err = gt_vec[4] - vec[4];
  Scalar y_err = gt_vec[5] - vec[5];
  lin_mse = (x_err*x_err) + (y_err*y_err);

  Scalar roll, pitch, yaw;
  toEulerAngles(vec[0], vec[1], vec[2], vec[3],
		roll, pitch, yaw);
  
  Scalar yaw_err = yaw - gt_vec[3];
  yaw_err = CppAD::atan2(CppAD::sin(yaw_err), CppAD::cos(yaw_err));
  ang_mse = CppAD::abs(yaw_err);
}


template<typename Scalar>
void VehicleSystem<Scalar>::integrate(const VectorS &Xk, VectorS &Xk1)
{
  Eigen::Matrix<Scalar, HybridDynamics::STATE_DIM, 1> model_x0;
  Eigen::Matrix<Scalar, HybridDynamics::CNTRL_DIM, 1> model_u;
  Eigen::Matrix<Scalar, HybridDynamics::STATE_DIM, 1> model_x1;
  
  for(int i = 0; i < model_x0.size(); i++)
  {
    model_x0[i] = Xk[i];
  }
  for(int i = 0; i < model_u.size(); i++)
  {
    model_u[i] = Xk[i+model_x0.size()];
  }
  
  
  const int num_steps = 10; // 10*.001 = .01
  for(int ii = 0; ii < num_steps; ii++)
  {
    model_x0[17] = model_x0[19] = model_u[0];
    model_x0[18] = model_x0[20] = model_u[1];
    
    m_hybrid_dynamics.RK4(model_x0, model_x1, model_u);
    model_x0 = model_x1;
  }  
  
  
  for(int i = 0; i < model_x1.size(); i++)
  {
    Xk1[i] = model_x1[i];
  }
  for(int i = 0; i < model_u.size(); i++)
  {
    Xk1[i+model_x1.size()] = 0;
  }  
}

template<typename Scalar>
void VehicleSystem<Scalar>::getDefaultInitialState(VectorS &state)
{
	state = VectorS::Zero(state.size());
  
	m_hybrid_dynamics.initState(); //set start pos to 0,0,.16 and orientation to 0,0,0,1
	m_hybrid_dynamics.settle();     //allow the 3d vehicle to come to rest and reach steady state, equillibrium sinkage for tires.

	for(int i = 0; i < this->getStateDim(); i++)
	{
		state[i] = m_hybrid_dynamics.state_[i];
	}

	state[4] = 0.0;
	state[5] = 0.0;
}

template<typename Scalar>
void VehicleSystem<Scalar>::getDefaultParams(VectorS &params)
{
  m_hybrid_dynamics.tire_network.load_model();
  getParams(params);
}

template<typename Scalar>
VehicleSystem<Scalar>::VectorS VehicleSystem<Scalar>::initializeState(const GroundTruthDataRow &gt_state)
{
	Scalar xk[this->getStateDim()];
	Scalar xk_base[this->getStateDim()];
	VectorS xk_robot(this->getStateDim() +
					 this->getControlDim());
	VectorS yaw_quat(4);  
  
	yaw_quat[0] = 0;
	yaw_quat[1] = 0;
	yaw_quat[2] = std::sin(gt_state.yaw / 2.0); // rotating by yaw around z axis
	yaw_quat[3] = std::cos(gt_state.yaw / 2.0); // https://stackoverflow.com/questions/4436764/rotating-a-quaternion-on-1-axis
	
	xk[0] = yaw_quat[0]; // Quaternion. Sets initial yaw.
	xk[1] = yaw_quat[1];
	xk[2] = yaw_quat[2];
	xk[3] = yaw_quat[3];
  
	xk[4] = gt_state.x; // Position
	xk[5] = gt_state.y;
	xk[6] = gt_state.z;

	xk[7] = 0; // Joint positions
	xk[8] = 0;
	xk[9] = 0;
	xk[10] = 0;

	xk[11] = 0; // Spatial Velocity
	xk[12] = 0;
	xk[13] = gt_state.wz;
	xk[14] = gt_state.vx;
	xk[15] = gt_state.vy;
	xk[16] = 0;

	xk[17] = 0; // Joint velocities
	xk[18] = 0;
	xk[19] = 0;
	xk[20] = 0;

	// Unfortunately, the state vector is not expressed at the COM. Depressing. So we must transform it
	m_hybrid_dynamics.initStateCOM(&xk[0], &xk_base[0]);
  
	for(int i = 0; i < this->getStateDim(); i++)
	{
		xk_robot[i] = xk_base[i];
	}
  
	xk_robot[21] = gt_state.vl; // Control tire velocities
	xk_robot[22] = gt_state.vr;

	return xk_robot;
}


template class VehicleSystem<ADF>;
//template class VehicleSystem<ADAD>;
//template class VehicleSystem<float>;
