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
			6.0208e-02, -2.4415e+00,  2.2706e+00,  6.1775e-03,  2.4264e-01,
			1.0056e+00, -1.7730e+00,  2.9975e-01, -2.9747e-01,  9.4275e-03,
			-7.3466e-04,  3.8031e-04,  1.2523e-02,  2.4082e+00, -2.3937e+00,
			-4.8467e-02,  2.6775e-01,  1.0684e-02, -7.3954e-03,  2.5190e-03,
			-9.2957e-02, -8.3676e-01,  3.3758e-01,  2.3556e-01, -3.3596e-01,
			5.9547e-01, -1.4545e-01,  2.6162e-02,  2.8245e-01, -4.5151e-01,
			-4.3081e-01, -7.7941e-01;
		
		m_params[kk].bias0 <<
			-1.7725e-03, -2.4909e+00, -5.3862e-01,  5.6017e-01, -1.1865e+00,
			-8.0090e-01,  1.0362e+00, -1.7953e+00;
			
		m_params[kk].weight2 <<
			-9.1216e-01,  1.0414e-02, -2.4043e-01, -9.8472e-01, -4.5437e-01,
			-2.7477e-01, -1.0470e+00,  1.4964e+00,  7.5325e-03, -1.8032e-03,
			7.5933e-01,  1.0014e-02, -1.2761e+00,  5.8871e-03,  1.6312e-02,
			5.4238e-03, -1.3098e+00,  8.9473e-01, -2.2393e-01, -6.6229e-01,
			-8.8675e-01,  1.8582e+00, -8.0630e-01,  3.4219e-01,  1.4058e+00,
			-7.2921e-01, -1.1952e+00,  1.9974e+00,  4.1152e-01,  1.1201e+00,
			-1.9313e-01,  1.2842e+00, -1.0677e+00, -1.5490e-01,  1.7552e-01,
			-1.3352e+00,  5.4020e-01,  1.9964e+00, -3.5099e-01, -9.9138e-01,
			-1.4758e-01,  4.4947e-02, -6.3618e-01, -1.5209e-01, -5.5414e-02,
			1.3186e-01, -1.3754e+00, -1.4841e-03,  1.0144e+00,  3.1377e-01,
			2.6908e-01,  1.3219e+00,  1.1810e+00, -2.0219e+00,  9.5090e-01,
			-1.4074e+00,  6.7491e-01, -5.0405e-01,  1.3185e+00, -1.7848e-01,
			4.0171e-01, -1.6048e+00, -1.6768e-01, -1.3084e+00;
			
		m_params[kk].bias2 <<
			0.7312, -0.4782, -0.7218, -1.3401, -0.5225,  1.2047, -0.5494,  0.1772;
			
		m_params[kk].weight4 <<
			1.1953,  0.5256, -1.7765,  4.3787, -4.1470,  3.8591, -2.6577,  0.2952,
			4.4380,  0.5216, -2.9671, -2.3300, -5.3883,  1.1109, -5.1804, -2.5124,
			-0.0093, -7.6857, -0.0827, -0.0283, -0.0100, -0.0672, -0.0190, -0.0692;
		
		m_params[kk].bias4 <<
			2.5150, 0.7096, 4.5564;
	}
    
	return 0;
}
