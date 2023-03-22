#include "TireNetwork.h"
#include <cmath>
#include <cppad/cppad.hpp>
#include <assert.h>

Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,TireNetwork::num_in_features>  TireNetwork::m_weight0;
Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,TireNetwork::num_hidden_nodes> TireNetwork::m_weight2;
Eigen::Matrix<Scalar,TireNetwork::num_out_features,TireNetwork::num_hidden_nodes> TireNetwork::m_weight4;

Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> TireNetwork::m_bias0;
Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> TireNetwork::m_bias2;
Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> TireNetwork::m_bias4;
Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> TireNetwork::m_out_mean;
Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> TireNetwork::m_out_std;
Eigen::Matrix<Scalar,TireNetwork::num_in_features,1>  TireNetwork::m_in_mean;
Eigen::Matrix<Scalar,TireNetwork::num_in_features,1>  TireNetwork::m_in_std_inv;


int TireNetwork::m_is_loaded = 0;

TireNetwork::TireNetwork(){
  if(!m_is_loaded){
    load_model();
  }
}
TireNetwork::~TireNetwork(){}

//type overloading problem when taking address of std::tanh
//there is probably a smarter way to do this with templates.
inline double tanh_double_wrapper(double x){
  return std::tanh(x);
}

inline Scalar tanh_scalar_wrapper(Scalar x){
  return CppAD::tanh(x);
}

void TireNetwork::forward(const Eigen::Matrix<Scalar,num_in_features,1> &in_vec, Eigen::Matrix<Scalar,num_out_features,1> &out_vec){
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> layer0_out;
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> layer2_out;
  Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> layer4_out;
  Eigen::Matrix<Scalar,TireNetwork::num_in_features,1> scaled_features;
  
  scaled_features = (in_vec - m_in_mean).cwiseProduct(m_in_std_inv);
  layer0_out = (m_weight0*scaled_features) + m_bias0;
  layer0_out = layer0_out.unaryExpr(&tanh_scalar_wrapper);
  layer2_out = (m_weight2*layer0_out) + m_bias2;
  layer2_out = layer2_out.unaryExpr(&tanh_scalar_wrapper);
  layer4_out = (m_weight4*layer2_out) + m_bias4;        
  out_vec = layer4_out.cwiseProduct(m_out_std) + m_out_mean;
}

int TireNetwork::getNumParams()
{
  return m_in_mean.size() +
    m_in_std_inv.size() +
    m_weight0.size() +
    m_bias0.size() +
    m_weight2.size() +
    m_bias2.size() +
    m_weight4.size() +
    m_bias4.size() +
    m_out_mean.size() +
    m_out_std.size();
}

void TireNetwork::setParams(const VectorS &params, int idx)
{
  assert(params.size() == getNumParams());
  for(int i = 0; i < m_weight0.rows(); i++)
  {
    for(int j = 0; j < m_weight0.cols(); j++)
    {
      m_weight0(i,j) = params[idx];
      idx++;
    }
  }
  for(int j = 0; j < m_bias0.size(); j++)
  {
    m_bias0[j] = params[idx];
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
  for(int j = 0; j < m_bias2.size(); j++)
  {
    m_bias2[j] = params[idx];
    idx++;
  }

  for(int i = 0; i < m_weight4.rows(); i++)
  {
    for(int j = 0; j < m_weight4.cols(); j++)
    {
      m_weight4(i,j) = params[idx];
      idx++;
    }
  }
  for(int j = 0; j < m_bias4.size(); j++)
  {
    m_bias4[j] = params[idx];
    idx++;
  }


  for(int j = 0; j < m_in_mean.size(); j++)
  {
    m_in_mean[j] = params[idx];
    idx++;
  }
  for(int j = 0; j < m_in_std_inv.size(); j++)
  {
    m_in_std_inv[j] = params[idx];
    idx++;
  }
  for(int j = 0; j < m_out_mean.size(); j++)
  {
    m_out_mean[j] = params[idx];
    idx++;
  }
  for(int j = 0; j < m_out_std.size(); j++)
  {
    m_out_std[j] = params[idx];
    idx++;
  }
}

void TireNetwork::getParams(VectorS &params, int idx)
{
  assert(params.size() == getNumParams());
  
  for(int i = 0; i < m_weight0.rows(); i++)
  {
    for(int j = 0; j < m_weight0.cols(); j++)
    {
      params[idx] = m_weight0(i,j);
      idx++;
    }
  }
  for(int j = 0; j < m_bias0.size(); j++)
  {
    params[idx] = m_bias0[j];
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
  for(int j = 0; j < m_bias2.size(); j++)
  {
    params[idx] = m_bias2[j];
    idx++;
  }

  for(int i = 0; i < m_weight4.rows(); i++)
  {
    for(int j = 0; j < m_weight4.cols(); j++)
    {
      params[idx] = m_weight4(i,j);
      idx++;
    }
  }
  for(int j = 0; j < m_bias4.size(); j++)
  {
    params[idx] = m_bias4[j];
    idx++;
  }


  for(int j = 0; j < m_in_mean.size(); j++)
  {
    params[idx] = m_in_mean[j];
    idx++;
  }
  for(int j = 0; j < m_in_std_inv.size(); j++)
  {
    params[idx] = m_in_std_inv[j];
    idx++;
  }
  for(int j = 0; j < m_out_mean.size(); j++)
  {
    params[idx] = m_out_mean[j];
    idx++;
  }
  for(int j = 0; j < m_out_std.size(); j++)
  {
    params[idx] = m_out_std[j];
    idx++;
  }  
}




#include "TireNetworkWeights.cpp"
