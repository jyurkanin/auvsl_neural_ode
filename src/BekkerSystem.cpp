#include "BekkerSystem.h"
#include "utils.h"

template<typename Scalar>
BekkerSystem<Scalar>::BekkerSystem(const std::shared_ptr<const TerrainMap<Scalar>> &map)
{
	this->setNumParams(5);
	this->setStateDim(BekkerDynamics::STATE_DIM);
	this->setControlDim(BekkerDynamics::CNTRL_DIM);

	m_bekker_dynamics.setTerrainMap(map);
}

template<typename Scalar>
BekkerSystem<Scalar>::~BekkerSystem()
{

}

template<typename Scalar>
void BekkerSystem<Scalar>::getDefaultParams(VectorS &params)
{
	params[0] = .2976;
	params[1] = 2.083;
	params[2] = 0.8;
	params[3] = 0.0;
	params[4] = .3927;
}

template<typename Scalar>
void BekkerSystem<Scalar>::getDefaultInitialState(VectorS &state)
{
	state = VectorS::Zero(state.size());
	
	m_bekker_dynamics.initState(); //set start pos to 0,0,.16 and orientation to 0,0,0,1
	m_bekker_dynamics.settle(); //allow the 3d vehicle to come to rest and reach steady state, equillibrium sinkage for tires.
	
	// Quaternion
	state[0] = m_hybrid_dynamics.state_[0];
	state[1] = m_hybrid_dynamics.state_[1];
	state[2] = m_hybrid_dynamics.state_[2];
	state[3] = m_hybrid_dynamics.state_[3];

	// Position
	state[4] = 0.0;
	state[5] = 0.0;
	state[6] = m_hybrid_dynamics.state_[6];
}


template<typename Scalar>
void BekkerSystem<Scalar>::setParams(const VectorS &params)
{
	m_bekker_dynamics.setParams(params);

	std::cout << "Params: "
			  << CppAD::Value(params[0]) << ", "
			  << CppAD::Value(params[1]) << ", "
			  << CppAD::Value(params[2]) << ", "
			  << CppAD::Value(params[3]) << ", "
			  << CppAD::Value(params[4]) << "\n";
}

template<typename Scalar>
void BekkerSystem<Scalar>::getParams(VectorS &params)
{
	m_bekker_dynamics.getParams(params);
}

template<typename Scalar>
void BekkerSystem<Scalar>::forward(const VectorS &X, VectorS &Xd)
{
	assert(X.size() == this->getStateDim());
	assert(Xd.size() == this->getStateDim());
  
	Eigen::Matrix<Scalar, BekkerDynamics::STATE_DIM, 1> model_x;
	Eigen::Matrix<Scalar, BekkerDynamics::CNTRL_DIM, 1> model_u;
	Eigen::Matrix<Scalar, BekkerDynamics::STATE_DIM, 1> model_xd;
  
	for(int i = 0; i < model_x.size(); i++)
	{
		model_x[i] = X[i];
	}
	for(int i = 0; i < model_u.size(); i++)
	{
		model_u[i] = X[i+model_x.size()];
	}
  
	m_bekker_dynamics.ODE(model_x, model_xd, model_u);
  
	for(int i = 0; i < model_xd.size(); i++)
	{
		Xd[i] = model_xd[i];
	}
	for(int i = 0; i < model_u.size(); i++)
	{
		Xd[i+model_x.size()] = 0;
	}
}

template<typename Scalar>
Scalar BekkerSystem<Scalar>::loss(const VectorS &gt_vec, VectorS &vec)
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
void BekkerSystem<Scalar>::evaluate(const VectorS &gt_vec, const VectorS &vec, Scalar &ang_mse, Scalar &lin_mse)
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
void BekkerSystem<Scalar>::integrate(const VectorS &Xk, VectorS &Xk1)
{
  Eigen::Matrix<Scalar, BekkerDynamics::STATE_DIM, 1> model_x0;
  Eigen::Matrix<Scalar, BekkerDynamics::CNTRL_DIM, 1> model_u;
  Eigen::Matrix<Scalar, BekkerDynamics::STATE_DIM, 1> model_x1;
  
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
    
    m_bekker_dynamics.RK4(model_x0, model_x1, model_u);
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
BekkerSystem<Scalar>::VectorS BekkerSystem<Scalar>::initializeState(const GroundTruthDataRow &gt_state)
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
	m_bekker_dynamics.initStateCOM(&xk[0], &xk_base[0]);
	
	for(int i = 0; i < this->getStateDim(); i++)
	{
		xk_robot[i] = xk_base[i];
	}
	
	xk_robot[21] = gt_state.vl; // Control tire velocities
	xk_robot[22] = gt_state.vr;
	
	return xk_robot;
}



template class BekkerSystem<ADF>;
