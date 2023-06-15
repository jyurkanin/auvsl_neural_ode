#include "VehicleSystem.h"
#include "utils.h"

#include <iostream>
#include <assert.h>

template<typename Scalar>
VehicleSystem<Scalar>::VehicleSystem() : cpp_bptt::System<Scalar>(HybridDynamics::STATE_DIM + HybridDynamics::CNTRL_DIM, 0)
{
  this->setNumParams(m_hybrid_dynamics.tire_network.getNumParams());
  this->setNumSteps(10);
  this->setTimestep(0.001);  //unused
  this->setLearningRate(1e-4f);
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
  Scalar x_err = gt_vec[4] - vec[4];
  Scalar y_err = gt_vec[5] - vec[5];
  Scalar lin_err = CppAD::sqrt((x_err*x_err) + (y_err*y_err));

  Scalar roll, pitch, yaw;
  toEulerAngles(vec[3], vec[0], vec[1], vec[2],
		roll, pitch, yaw);
  
  Scalar yaw_err = yaw - gt_vec[3];
  yaw_err = CppAD::atan2(CppAD::sin(yaw_err), CppAD::cos(yaw_err));
  Scalar ang_err = CppAD::abs(yaw_err);
  
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
  ang_mse = yaw_err*yaw_err;
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
  state = VectorS::Zero(this->getStateDim());
  state[3] = Scalar(1.0);
  state[6] = Scalar(0.16);
}

template<typename Scalar>
void VehicleSystem<Scalar>::getDefaultParams(VectorS &params)
{
  m_hybrid_dynamics.tire_network.load_model();
  getParams(params);
}

template class VehicleSystem<ADF>;
//template class VehicleSystem<ADAD>;
//template class VehicleSystem<float>;
