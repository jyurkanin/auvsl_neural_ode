#include "BaseNetwork.h"

// Haxx
inline Scalar tanh_wrapper(Scalar x){
  return CppAD::tanh(x);
}

inline Scalar relu_wrapper(Scalar x){
  Scalar zero{0};
  return CppAD::CondExpGt(x, zero, x, zero);
}


BaseNetwork::BaseNetwork()
{
	m_weight0 = 0.001*Eigen::Matrix<Scalar,m_num_hidden_units,m_num_inputs>::Random();
	m_bias0   = 0.001*Eigen::Matrix<Scalar,m_num_hidden_units,1>::Random();
	m_weight1 = 0.001*Eigen::Matrix<Scalar,m_num_hidden_units,m_num_hidden_units>::Random();
	m_bias1   = 0.001*Eigen::Matrix<Scalar,m_num_hidden_units,1>::Random();
	m_weight2 = 0.001*Eigen::Matrix<Scalar,m_num_outputs,m_num_hidden_units>::Random();
	m_bias2   = 0.001*Eigen::Matrix<Scalar,m_num_outputs,1>::Random();
	
	out_std    = 10.0*Eigen::Matrix<Scalar,m_num_outputs,1>::Ones();
	in_std_inv = 0.1*Eigen::Matrix<Scalar,m_num_inputs,1>::Ones();
}

BaseNetwork::~BaseNetwork(){}

int BaseNetwork::getNumParams()
{
	return m_weight0.size() +
		   m_bias0.size() +
		   m_weight1.size() +
		   m_bias1.size() +
		   m_weight2.size() +
		   m_bias2.size();
}

// in_vec: [wz, vx, vy]
// out_vec: [nz, fx, fy]
void BaseNetwork::forward(const Eigen::Matrix<Scalar,m_num_inputs,1> &in_vec,
						  Eigen::Matrix<Scalar,m_num_outputs,1> &out_vec)
{
	Eigen::Matrix<Scalar,m_num_hidden_units,1> layer0;
	Eigen::Matrix<Scalar,m_num_hidden_units,1> layer1;
	Eigen::Matrix<Scalar,m_num_outputs,1>      layer2;
	Eigen::Matrix<Scalar,m_num_inputs,1>       scaled;

	scaled = in_vec.cwiseProduct(in_std_inv);
	
	layer0 = (m_weight0*scaled) + m_bias0; layer0 = layer0.unaryExpr(&tanh_wrapper);
	layer1 = (m_weight1*layer0) + m_bias1; layer1 = layer1.unaryExpr(&tanh_wrapper);
	layer2 = (m_weight2*layer1) + m_bias2;
	
	layer2[0] = relu_wrapper(layer2[0])*(-in_vec[0]);
	layer2[1] = relu_wrapper(layer2[1])*(-in_vec[1]);
	layer2[2] = relu_wrapper(layer2[2])*(-in_vec[2]);
	
	out_vec = layer2.cwiseProduct(out_std);
}

void BaseNetwork::setParams(const VectorS &params, int idx)
{
	for(int i = 0; i < m_weight0.rows(); i++)
	{
		for(int j = 0; j < m_weight0.cols(); j++)
		{
			m_weight0(i,j) = params[idx];
			idx++;
		}
	}
	for(int i = 0; i < m_bias0.size(); i++)
	{
		m_bias0[i] = params[idx];
		idx++;
	}
	
	for(int i = 0; i < m_weight1.rows(); i++)
	{
		for(int j = 0; j < m_weight1.cols(); j++)
		{
			m_weight1(i,j) = params[idx];
			idx++;
		}
	}
	for(int i = 0; i < m_bias1.size(); i++)
	{
		m_bias1[i] = params[idx];
		idx++;
	}

	
	for(int i = 0; i < m_weight2.rows(); i++)
	{
		for(int j = 0; j < m_weight2.cols(); j++)
		{
			m_weight2(i,j) = params[idx];
			idx++;
		}
	}
	for(int i = 0; i < m_bias2.size(); i++)
	{
		m_bias2[i] = params[idx];
		idx++;
	}

}



void BaseNetwork::getParams(VectorS &params, int idx)
{
	for(int i = 0; i < m_weight0.rows(); i++)
	{
		for(int j = 0; j < m_weight0.cols(); j++)
		{
			params[idx] = m_weight0(i,j);
			idx++;
		}
	}
	for(int i = 0; i < m_bias0.size(); i++)
	{
		params[idx] = m_bias0[i];
		idx++;
	}
	
	for(int i = 0; i < m_weight1.rows(); i++)
	{
		for(int j = 0; j < m_weight1.cols(); j++)
		{
			params[idx] = m_weight1(i,j);
			idx++;
		}
	}
	for(int i = 0; i < m_bias1.size(); i++)
	{
		params[idx] = m_bias1[i];
		idx++;
	}

	
	for(int i = 0; i < m_weight2.rows(); i++)
	{
		for(int j = 0; j < m_weight2.cols(); j++)
		{
			params[idx] = m_weight2(i,j);
			idx++;
		}
	}
	for(int i = 0; i < m_bias2.size(); i++)
	{
		params[idx] = m_bias2[i];
		idx++;
	}
}
