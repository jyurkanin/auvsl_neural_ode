#include "linear/LinearSystem.h"

#include <iostream>
#include <assert.h>


LinearSystem::LinearSystem()
{
	setNumParams(6);
	setStateDim(3);
	setControlDim(2);

	m_params = MatrixAD::Random(3,2);
	m_params(0,0) = 0.0472;
	m_params(1,0) = 0.0036;
	m_params(2,0) = -.1744;
	
	m_params(0,1) = 0.0438;
	m_params(1,1) = -0.0035;
	m_params(2,1) = 0.1749;	
}


LinearSystem::~LinearSystem()
{
  
}


void LinearSystem::setParams(const VectorAD &params)
{
	m_params(0,0) = params[0];
	m_params(1,0) = params[1];
	m_params(2,0) = params[2];
	m_params(0,1) = params[3];
	m_params(1,1) = params[4];
	m_params(2,1) = params[5];
}


void LinearSystem::getParams(VectorAD &params)
{
	params[0] = m_params(0,0);
	params[1] = m_params(1,0);
	params[2] = m_params(2,0);
	params[3] = m_params(0,1);
	params[4] = m_params(1,1);
	params[5] = m_params(2,1);
}



ADF LinearSystem::loss(const VectorAD &gt_vec, VectorAD &vec)
{
	ADF vx_err = (gt_vec[0] - vec[0]);
	ADF vy_err = (gt_vec[1] - vec[1]);
	ADF wz_err = (gt_vec[2] - vec[2]);
	return (wz_err*wz_err) + (vx_err*vx_err) + (vy_err*vy_err);
}


void LinearSystem::evaluate(const VectorAD &gt_vec, const VectorAD &vec, ADF &ang_mse, ADF &lin_mse)
{
	ADF x_err = gt_vec[0] - vec[0];
	ADF y_err = gt_vec[1] - vec[1];
	lin_mse = (x_err*x_err) + (y_err*y_err);
	
	ADF yaw_err = gt_vec[2] - vec[2];
	yaw_err = CppAD::atan2(CppAD::sin(yaw_err), CppAD::cos(yaw_err));
	ang_mse = yaw_err*yaw_err;
}


void LinearSystem::forward(const VectorAD &u, VectorAD &Xd)
{
	assert(u.size() == 2);
	assert(Xd.size() == 3);
		
	Xd = m_params * u;
}


void LinearSystem::integrate(const VectorAD &Xk, VectorAD &Xk1)
{	
	VectorAD u(2);
	VectorAD Xd_w(3);
	VectorAD Xd_b(3);
	u[0] = Xk[3];
 	u[1] = Xk[4];
	forward(u, Xd_b);
	
	Xd_w[0] = CppAD::cos(Xk[2])*Xd_b[0] - CppAD::sin(Xk[2])*Xd_b[1];
	Xd_w[1] = CppAD::sin(Xk[2])*Xd_b[0] + CppAD::cos(Xk[2])*Xd_b[1];	
	Xd_w[2] = Xd_b[2];
	
	Xk1[0] = Xk[0] + Xd_w[0]*m_timestep;
	Xk1[1] = Xk[1] + Xd_w[1]*m_timestep;
	Xk1[2] = Xk[2] + Xd_w[2]*m_timestep;
	
	Xk1[3] = 0;
 	Xk1[4] = 0;
}


void LinearSystem::getDefaultInitialState(VectorAD &state)
{
	state[0] = 0;
 	state[1] = 0;
	state[2] = 0;
}

void LinearSystem::getDefaultParams(VectorAD &params)
{
	params[0] = 0.0542;
	params[1] = 0.0036;
	params[2] = -0.1743;
	
	params[3] = 0.0370;
	params[4] = -0.0035;
	params[5] = 0.1749;
}

