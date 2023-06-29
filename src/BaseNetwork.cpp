#include "BaseNetwork.h"

BaseNetwork::BaseNetwork()
{
	m_weight0 = Eigen::Matrix<Scalar,m_num_hidden_units,m_num_inputs>::Random();
	m_bias0   = Eigen::Matrix<Scalar,m_num_hidden_units,1>::Random();
	m_weight1 = Eigen::Matrix<Scalar,m_num_hidden_units,m_num_hidden_units>::Random();
	m_bias1   = Eigen::Matrix<Scalar,m_num_hidden_units,1>::Random();
	m_weight2 = Eigen::Matrix<Scalar,m_num_outputs,m_num_hidden_units>::Random();
	m_bias2   = Eigen::Matrix<Scalar,m_num_outputs,1>::Random();
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

void setParams(const VectorS &params, int idx)
{
	for(int i = 0; i < m_weight0.rows(); i++)
	{
		for(int j = 0; j < m_weight0.cols(); j++)
		{
			m_weight0(i,j) = params[idx];
			idx++
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
			idx++
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
			idx++
		}
	}
	for(int i = 0; i < m_bias2.size(); i++)
	{
		m_bias2[i] = params[idx];
		idx++;
	}

}



void getParams(VectorS &params, int idx)
{
	for(int i = 0; i < m_weight0.rows(); i++)
	{
		for(int j = 0; j < m_weight0.cols(); j++)
		{
			params[idx] = m_weight0(i,j);
			idx++
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
			idx++
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
			idx++
		}
	}
	for(int i = 0; i < m_bias2.size(); i++)
	{
		params[idx] = m_bias2[i];
		idx++;
	}
}
