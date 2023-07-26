#pragma once

#include <cpp_bptt.h>
#include "HybridDynamics.h"

// I believe this is ready.
// This represents an ODE and loss function

template<typename Scalar>
class VehicleSystem : public cpp_bptt::System<Scalar>
{
public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;
  
  VehicleSystem();
  ~VehicleSystem();

  virtual void   getDefaultParams(VectorS &params);
  virtual void   getDefaultInitialState(VectorS &state);
  virtual void   setParams(const VectorS &params);
  virtual void   getParams(VectorS &params);
  virtual void   forward(const VectorS &X, VectorS &Xd);
  virtual Scalar loss(const VectorS &gt_vec, VectorS &vec);
  
  void evaluate(const VectorS &gt_vec, const VectorS &vec, Scalar &ang_err, Scalar &lin_err);
  void integrate(const VectorS &X0, VectorS &X1);  
  void reset();
  Scalar getPenalty();
	
  Scalar m_penalty_weight;
  MatrixS m_params;
  HybridDynamics m_hybrid_dynamics;
};
