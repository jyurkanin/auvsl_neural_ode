#pragma once

#include <Eigen/Dense>
#include "generated/forward_dynamics.h"

using Jackal::rcg::Scalar;

// Feedforward network 2 hidden layers.
// Inputs: Vx, Vy, Wz, 
// Outputs: Fx, Fy, Nz
class BaseNetwork
{
public:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;

	BaseNetwork();
	~BaseNetwork();

	static const int m_num_hidden_units = 8;
	static const int m_num_inputs = 3;
	static const int m_num_outputs = 3;
	
	int getNumInputs() const { return m_num_inputs; }
	int getNumOutputs() const { return m_num_outputs; }
	
	int getNumParams();
	void setParams(const VectorS &params, int idx);
	void getParams( VectorS &params, int idx);
	void forward(const Eigen::Matrix<Scalar,m_num_inputs,1> &in_vec, Eigen::Matrix<Scalar,m_num_outputs,1> &out_vec);

	Eigen::Matrix<Scalar,m_num_hidden_units,m_num_inputs> m_weight0;
	Eigen::Matrix<Scalar,m_num_hidden_units,1> m_bias0;
	Eigen::Matrix<Scalar,m_num_hidden_units,m_num_hidden_units> m_weight1;
	Eigen::Matrix<Scalar,m_num_hidden_units,1> m_bias1;
	Eigen::Matrix<Scalar,m_num_outputs,m_num_hidden_units> m_weight2;
	Eigen::Matrix<Scalar,m_num_outputs,1> m_bias2;

	
private:
	// You're just going to guess this shit. Not gonna lie. Fuck it.
	// Dont feel like doing a norm layer.
	Eigen::Matrix<Scalar,m_num_outputs,1> out_std;
	Eigen::Matrix<Scalar,m_num_inputs,1> in_std_inv;
};
