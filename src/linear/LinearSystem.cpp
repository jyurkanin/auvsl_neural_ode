#include "linear/LinearSystem.h"
#include "utils.h"

#include <iostream>
#include <assert.h>

template<typename Scalar>
LinearSystem<Scalar>::LinearSystem() : cpp_bptt::System<Scalar>(5, 0)
{
	this->setNumParams(6);
	this->setNumSteps(10);
	this->setTimestep(1e-2f); // unused
	this->setLearningRate(1e-3f);

	m_params = MatrixS::Random(3,2);
	m_params(0,0) = 0.0472;
	m_params(1,0) = 0.0036;
	m_params(2,0) = -.1744;
	
	m_params(0,1) = 0.0438;
	m_params(1,1) = -0.0035;
	m_params(2,1) = 0.1749;	
}

template<typename Scalar>
LinearSystem<Scalar>::~LinearSystem()
{
  
}

template<typename Scalar>
void LinearSystem<Scalar>::setParams(const VectorS &params)
{
	m_params(0,0) = params[0];
	m_params(1,0) = params[1];
	m_params(2,0) = params[2];
	m_params(0,1) = params[3];
	m_params(1,1) = params[4];
	m_params(2,1) = params[5];
}

template<typename Scalar>
void LinearSystem<Scalar>::getParams(VectorS &params)
{
	params[0] = m_params(0,0);
	params[1] = m_params(1,0);
	params[2] = m_params(2,0);
	params[3] = m_params(0,1);
	params[4] = m_params(1,1);
	params[5] = m_params(2,1);
}


template<typename Scalar>
Scalar LinearSystem<Scalar>::loss(const VectorS &gt_vec, VectorS &vec)
{
	Scalar vx_err = (gt_vec[0] - vec[0]);
	Scalar vy_err = (gt_vec[1] - vec[1]);
	Scalar wz_err = (gt_vec[2] - vec[2]);
	return (wz_err*wz_err) + (vx_err*vx_err) + (vy_err*vy_err);
}

template<typename Scalar>
void LinearSystem<Scalar>::evaluate(const VectorS &gt_vec, const VectorS &vec, Scalar &ang_mse, Scalar &lin_mse)
{
	Scalar x_err = gt_vec[0] - vec[0];
	Scalar y_err = gt_vec[1] - vec[1];
	lin_mse = (x_err*x_err) + (y_err*y_err);
	
	Scalar yaw_err = gt_vec[2] - vec[2];
	yaw_err = CppAD::atan2(CppAD::sin(yaw_err), CppAD::cos(yaw_err));
	ang_mse = yaw_err*yaw_err;
}

template<typename Scalar>
void LinearSystem<Scalar>::forward(const VectorS &u, VectorS &Xd)
{
	assert(u.size() == 2);
	assert(Xd.size() == 3);
		
	Xd = m_params * u;
}

template<typename Scalar>
void LinearSystem<Scalar>::integrate(const VectorS &Xk, VectorS &Xk1)
{
	assert(Xk.size() == this->getStateDim());
	assert(Xk1.size() == this->getStateDim());
	
	VectorS u(2);
	VectorS Xd_w(3);
	VectorS Xd_b(3);
	u[0] = Xk[3];
 	u[1] = Xk[4];
	forward(u, Xd_b);
	
	Xd_w[0] = CppAD::cos(Xk[2])*Xd_b[0] - CppAD::sin(Xk[2])*Xd_b[1];
	Xd_w[1] = CppAD::sin(Xk[2])*Xd_b[0] + CppAD::cos(Xk[2])*Xd_b[1];	
	Xd_w[2] = Xd_b[2];
	
	Xk1[0] = Xk[0] + Xd_w[0]*this->getTimestep();
	Xk1[1] = Xk[1] + Xd_w[1]*this->getTimestep();
	Xk1[2] = Xk[2] + Xd_w[2]*this->getTimestep();
	
	Xk1[3] = 0;
 	Xk1[4] = 0;
}

template<typename Scalar>
void LinearSystem<Scalar>::getDefaultInitialState(VectorS &state)
{
	state[0] = 0;
 	state[1] = 0;
	state[2] = 0;
}

template<typename Scalar>
void LinearSystem<Scalar>::getDefaultParams(VectorS &params)
{
	params[0] = 0.0542;
	params[1] = 0.0036;
	params[2] = -0.1743;
	
	params[3] = 0.0370;
	params[4] = -0.0035;
	params[5] = 0.1749;
}


template class LinearSystem<ADF>;

