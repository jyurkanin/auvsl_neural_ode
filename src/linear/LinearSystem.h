#pragma once

#include "types/Scalars.h"
#include "utils.h"
#include <Eigen/Dense>

// I believe this is ready.
// This represents an ODE and loss function

///todo: implement the other base class functions
class LinearSystem
{
public:
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, 1> VectorAD;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> MatrixAD;
	
	LinearSystem();
	~LinearSystem();

	void getDefaultParams(VectorAD &params);
	void getDefaultInitialState(VectorAD &state);
	void setParams(const VectorAD &params);
	void getParams(VectorAD &params);

	/// This function represents the model Xd = B*u
	/// Xd is velocities in body coordinates
	void forward(const VectorAD &u, VectorAD &Xd);
	ADF  loss(const VectorAD &gt_vec, VectorAD &vec);

	void evaluate(const VectorAD &gt_vec, const VectorAD &vec, ADF &ang_err, ADF &lin_err);
	void integrate(const VectorAD &X0, VectorAD &X1);  

	int getStateDim(){return m_state_dim;}
	int getControlDim(){return m_control_dim;}
	int getNumParams(){return m_num_params;}
	void setNumParams(int num_params){m_num_params = num_params;}
	void setStateDim(int state_dim){m_state_dim = state_dim;}
	void setControlDim(int control_dim){m_control_dim = control_dim;}
	
private:
	int m_num_params;
	int m_state_dim;
	int m_control_dim;
	
	MatrixAD m_params;
	const ADF m_timestep{1e-2};
};
