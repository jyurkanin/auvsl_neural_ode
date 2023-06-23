#include "TireNetwork.h"
#include "generated/model_constants.h"
#include <cmath>
#include <cppad/cppad.hpp>
#include <iostream>



TireNetwork::TireNetwork()
{
  out_std << 44.45929437637364, 44.319044796830426, 55.11481922955709;
  in_mean << 0.005, 0.0, 0.0, 0.0;
  in_std_inv << 0.0028585679829120636, 0.5799984931945801, 0.576934278011322, 0.5774632096290588;
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


// vx vy w zr
void TireNetwork::forward(const Eigen::Matrix<Scalar,8,1> &in_vec, Eigen::Matrix<Scalar,num_out_features,1> &out_vec, int ii){
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> layer0_out;
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> layer2_out;
  Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> layer4_out;
  Eigen::Matrix<Scalar,TireNetwork::num_in_features,1> scaled_features;
  Eigen::Matrix<Scalar,TireNetwork::num_in_features,1> bekker_vec;
  
  // Changes features to cross the origin
  Scalar tire_tangent_vel = in_vec[2] * Jackal::rcg::tire_radius;
  Scalar diff = tire_tangent_vel - in_vec[0];
  Scalar slip_lon = CppAD::abs(diff);
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
			3.4663e-01, -6.4281e-01, -2.2576e-01, -2.8411e+00,  3.5161e-01,
			-1.9375e+00, -3.0442e-01, -2.1163e+00,  3.2830e-01, -1.3443e+00,
			-3.5285e-01,  1.1989e-01, -1.9947e-01,  9.1818e-04, -5.6239e-04,
			3.4403e-03, -2.2007e-01, -6.3835e-01, -1.0988e-04, -3.9429e-01,
			-2.2909e-01,  5.0217e+00, -5.8203e-01,  2.2315e-01, -1.7623e-01,
			8.8314e-01,  1.1601e-01,  4.3430e-01, -2.9912e-01,  4.2280e-01,
			1.8405e-01, -2.2359e-01;
		m_params[kk].bias0 <<
			-0.4173,  0.1699, -0.3258,  0.2898, -0.6835,  0.3161, -0.3273,  0.9159;
    
		m_params[kk].weight2 <<
			0.6797, -1.3332,  0.3648,  1.0987,  0.9242,  0.2478, -0.5886, -0.5536,
			-0.1417,  0.1988, -0.0503,  1.8130,  0.3068, -0.0117, -0.1186, -0.0075,
			-0.1201, -1.0499,  0.7991,  1.5475, -0.1323, -1.0842,  0.7463,  0.0955,
			-0.8544, -0.1083, -0.2443, -2.0554, -0.2436,  0.0116, -0.5936, -0.0456,
			-0.4345, -0.1200, -0.1570, -1.4308,  0.3886, -1.3547, -0.4473,  0.7988,
			0.4615, -0.8795, -0.2451, -0.1931,  0.2229,  0.4337,  0.7652,  0.2406,
			-0.8435,  4.1389, -0.3045,  1.6753,  0.5670, -1.4004, -1.3435,  1.1344,
			-0.2899,  0.1608,  0.0907, -2.3174, -2.5551,  0.2699,  0.9221,  0.1850;
		m_params[kk].bias2 <<
			-0.3069,  0.2451, -0.0706,  0.7074,  0.4181, -0.7531, -0.1848,  0.9071;
    
		m_params[kk].weight4 <<
			-5.4844,  3.1292, -2.5546,  2.1478,  3.2867, -4.0153,  4.5943,  3.2404,
			-3.5203,  0.9629, -3.0376,  1.8592, -1.1027, -4.7971,  3.1608,  2.0539,
			-0.3680, -5.0785, -0.4100,  2.5776,  0.0524, -0.6301,  0.1623,  1.8662;
		m_params[kk].bias4 <<
			1.7189, 2.1807, 1.3289;
	}
    
	return 0;
}
