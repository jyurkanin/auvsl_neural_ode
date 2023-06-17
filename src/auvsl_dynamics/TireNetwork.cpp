#include "TireNetwork.h"
#include "generated/model_constants.h"
#include <cmath>
#include <cppad/cppad.hpp>
#include <iostream>



TireNetwork::TireNetwork()
{
  out_std << 33.23321612383274, 26.304579611075567, 54.48882887004447;
  in_mean << 0.00505, 1.0, 5.0, 0.0;
  in_std_inv << 0.0028585679829120636, 0.5772117376327515, 2.884671449661255, 0.5774632096290588;
  in_std_inv = in_std_inv.cwiseInverse();  
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

inline Scalar relu_wrapper(Scalar x){
  Scalar zero{0};
  return CppAD::CondExpGt(x, zero, x, zero);
}


// vx vy w zr wz
void TireNetwork::forward(const Eigen::Matrix<Scalar,8,1> &in_vec, Eigen::Matrix<Scalar,num_out_features,1> &out_vec, int ii){
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> layer0_out;
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> layer2_out;
  Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> layer4_out;
  Eigen::Matrix<Scalar,TireNetwork::num_in_features,1> scaled_features;
  Eigen::Matrix<Scalar,TireNetwork::num_in_features,1> bekker_vec;
  
  // Changes features to cross the origin
  Scalar tire_tangent_vel = in_vec[2] * Jackal::rcg::tire_radius;
  Scalar diff = tire_tangent_vel - in_vec[0];
  Scalar slip_lon = CppAD::abs(in_vec[0]);
  Scalar slip_lat = CppAD::abs(in_vec[1]);
  Scalar tire_abs = CppAD::abs(in_vec[2]);
  
  bekker_vec[0] = in_vec[3];
  bekker_vec[1] = slip_lon;
  bekker_vec[2] = tire_abs;
  bekker_vec[3] = slip_lat;
  
  // Apply scaling after calculating the bekker features from kinematics
  scaled_features = (bekker_vec - in_mean).cwiseProduct(in_std_inv);
  
  // Actual NN math
  layer0_out = (m_params[ii].weight0*scaled_features) + m_params[ii].bias0;
  layer0_out = layer0_out.unaryExpr(&tanh_scalar_wrapper);
  layer2_out = (m_params[ii].weight2*layer0_out) + m_params[ii].bias2;
  layer2_out = layer2_out.unaryExpr(&tanh_scalar_wrapper);
  layer4_out = (m_params[ii].weight4*layer2_out) + m_params[ii].bias4;
  
  // Sign change passivity haxx
  out_vec[0] = relu_wrapper(layer4_out[0])*(1*diff);
  out_vec[1] = relu_wrapper(layer4_out[1])*(-1*in_vec[1]);
  out_vec[2] = relu_wrapper(layer4_out[2])/(1 + CppAD::exp(-1*in_vec[3]));
  
  // Scale output
  out_vec = out_vec.cwiseProduct(out_std);
}


int TireNetwork::getNumParams()
{
  return 4*(m_params[0].weight0.size() +
	    m_params[0].bias0.size() +
	    m_params[0].weight2.size() +
	    m_params[0].bias2.size() +
	    m_params[0].weight4.size() +
	    m_params[0].bias4.size());
}

void TireNetwork::setParams(const VectorS &params, int idx)
{
	assert(params.size() == getNumParams());
  
	for(int kk = 0; kk < num_networks; kk++)
	{
		for(int i = 0; i < m_params[kk].weight0.rows(); i++)
		{
			for(int j = 0; j < m_params[kk].weight0.cols(); j++)
			{
				m_params[kk].weight0(i,j) = params[idx];
				idx++;
			}
		}
		for(int j = 0; j < m_params[kk].bias0.size(); j++)
		{
			m_params[kk].bias0[j] = params[idx];
			idx++;
		}
  
		for(int i = 0; i < m_params[kk].weight2.rows(); i++)
		{
			for(int j = 0; j < m_params[kk].weight2.cols(); j++)
			{
				m_params[kk].weight2(i,j) = params[idx];
				idx++;
			}
		}
		for(int j = 0; j < m_params[kk].bias2.size(); j++)
		{
			m_params[kk].bias2[j] = params[idx];
			idx++;
		}

		for(int i = 0; i < m_params[kk].weight4.rows(); i++)
		{
			for(int j = 0; j < m_params[kk].weight4.cols(); j++)
			{
				m_params[kk].weight4(i,j) = params[idx];
				idx++;
			}
		}
		for(int j = 0; j < m_params[kk].bias4.size(); j++)
		{
			m_params[kk].bias4[j] = params[idx];
			idx++;
		}
	}
}

void TireNetwork::getParams(VectorS &params, int idx)
{
	assert(params.size() == getNumParams());

	for(int kk = 0; kk < num_networks; kk++)
	{
		for(int i = 0; i < m_params[kk].weight0.rows(); i++)
		{
			for(int j = 0; j < m_params[kk].weight0.cols(); j++)
			{
				params[idx] = m_params[kk].weight0(i,j);
				idx++;
			}
		}
		for(int j = 0; j < m_params[kk].bias0.size(); j++)
		{
			params[idx] = m_params[kk].bias0[j];
			idx++;
		}
  
		for(int i = 0; i < m_params[kk].weight2.rows(); i++)
		{
			for(int j = 0; j < m_params[kk].weight2.cols(); j++)
			{
				params[idx] = m_params[kk].weight2(i,j);
				idx++;
			}
		}
		for(int j = 0; j < m_params[kk].bias2.size(); j++)
		{
			params[idx] = m_params[kk].bias2[j];
			idx++;
		}

		for(int i = 0; i < m_params[kk].weight4.rows(); i++)
		{
			for(int j = 0; j < m_params[kk].weight4.cols(); j++)
			{
				params[idx] = m_params[kk].weight4(i,j);
				idx++;
			}
		}
		for(int j = 0; j < m_params[kk].bias4.size(); j++)
		{
			params[idx] = m_params[kk].bias4[j];
			idx++;
		}
	}
}



int TireNetwork::load_model(){
	std::cout << "Loading Model\n";
    
	for(int kk = 0; kk < num_networks; kk++)
	{
		m_params[kk].weight0 <<
			0.2648, -0.0012, -0.0008,  0.0051,  0.2788, -0.2646, -0.0841, -0.2142;		
		m_params[kk].bias0 <<
			-0.8361, -1.6538;
		
		m_params[kk].weight2 <<
			1.9573, -0.2230,  0.7678, -2.0080;	
		m_params[kk].bias2 <<
			1.5879, -1.0999;
			
		m_params[kk].weight4 <<
			5.1173,  -6.5648,   6.6290, -12.6530,   9.9240,  -0.5652;
		m_params[kk].bias4 <<
			1.0013,  1.9399, -1.1643;
	}
    
	return 0;
}
