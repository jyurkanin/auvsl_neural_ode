#include "TireNetwork.h"
#include "generated/model_constants.h"
#include <cmath>
#include <cppad/cppad.hpp>
#include <iostream>

TireNetwork::Params TireNetwork::m_params[4];


Scalar TireNetwork::vx_std;
Scalar TireNetwork::vx_mean;
Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> TireNetwork::out_std;
Eigen::Matrix<Scalar,TireNetwork::num_in_features,1>  TireNetwork::in_mean;
Eigen::Matrix<Scalar,TireNetwork::num_in_features,1>  TireNetwork::in_std_inv;


int TireNetwork::is_loaded = 0;


TireNetwork::TireNetwork(){
  if(!is_loaded){
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

inline Scalar relu_wrapper(Scalar x){
  Scalar zero{0};
  return CppAD::CondExpGt(x, zero, x, zero);
}


// vx vy w zr wz
void TireNetwork::forward(const Eigen::Matrix<Scalar,9,1> &in_vec, Eigen::Matrix<Scalar,num_out_features,1> &out_vec, int ii){
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
  bekker_vec[4] = in_vec[4];
  
  // Apply scaling after calculating the bekker features from kinematics
  scaled_features = (bekker_vec - in_mean).cwiseProduct(in_std_inv);

  // Actual NN math
  layer0_out = (m_params[ii].weight0*scaled_features) + m_params[ii].bias0;
  layer0_out = layer0_out.unaryExpr(&tanh_scalar_wrapper);
  layer2_out = (m_params[ii].weight2*layer0_out) + m_params[ii].bias2;
  layer2_out = layer2_out.unaryExpr(&tanh_scalar_wrapper);
  layer4_out = (m_params[ii].weight4*layer2_out) + m_params[ii].bias4;
  
  
  Eigen::Matrix<Scalar,1,1> vx_vec;
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes2,1> vx_layer0_out;
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes2,1> vx_layer2_out;
  Eigen::Matrix<Scalar,1,1> vx_layer4_out;
  
  vx_vec[0] = (CppAD::abs(in_vec[0]) - vx_mean) / vx_std;
  
  vx_layer0_out = (m_params[ii].vx_weight0*vx_vec) + m_params[ii].vx_bias0;
  vx_layer0_out = vx_layer0_out.unaryExpr(&tanh_scalar_wrapper);
  vx_layer2_out = (m_params[ii].vx_weight2*vx_layer0_out) + m_params[ii].vx_bias2;
  vx_layer2_out = vx_layer2_out.unaryExpr(&tanh_scalar_wrapper);
  vx_layer4_out = (m_params[ii].vx_weight4*vx_layer2_out) + m_params[ii].vx_bias4;
  
  // Sign change passivity haxx
  out_vec[0] =
    (relu_wrapper(layer4_out[0])*(1*diff)) +
    (relu_wrapper(vx_layer4_out[0])*(-in_vec[0]));
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
	    m_params[0].bias4.size() +
	    m_params[0].vx_weight0.size() +
	    m_params[0].vx_bias0.size() +
	    m_params[0].vx_weight2.size() +
	    m_params[0].vx_bias2.size() +
	    m_params[0].vx_weight4.size() +
	    m_params[0].vx_bias4.size());
	    
}

void TireNetwork::setParams(const VectorS &params, int idx)
{
  assert(params.size() == getNumParams());

  for(int kk = 0; kk < 4; kk++)
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


    
    for(int i = 0; i < m_params[kk].vx_weight0.rows(); i++)
    {
      for(int j = 0; j < m_params[kk].vx_weight0.cols(); j++)
      {
	m_params[kk].vx_weight0(i,j) = params[idx];
	idx++;
      }
    }
    for(int j = 0; j < m_params[kk].vx_bias0.size(); j++)
    {
      m_params[kk].vx_bias0[j] = params[idx];
      idx++;
    }
  
    for(int i = 0; i < m_params[kk].vx_weight2.rows(); i++)
    {
      for(int j = 0; j < m_params[kk].vx_weight2.cols(); j++)
      {
	m_params[kk].vx_weight2(i,j) = params[idx];
	idx++;
      }
    }
    for(int j = 0; j < m_params[kk].vx_bias2.size(); j++)
    {
      m_params[kk].vx_bias2[j] = params[idx];
      idx++;
    }

    for(int i = 0; i < m_params[kk].vx_weight4.rows(); i++)
    {
      for(int j = 0; j < m_params[kk].vx_weight4.cols(); j++)
      {
	m_params[kk].vx_weight4(i,j) = params[idx];
	idx++;
      }
    }
    for(int j = 0; j < m_params[kk].vx_bias4.size(); j++)
    {
      m_params[kk].vx_bias4[j] = params[idx];
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



    for(int i = 0; i < m_params[kk].vx_weight0.rows(); i++)
      {
	for(int j = 0; j < m_params[kk].vx_weight0.cols(); j++)
	  {
	    params[idx] = m_params[kk].vx_weight0(i,j);
	    idx++;
	  }
      }
    for(int j = 0; j < m_params[kk].vx_bias0.size(); j++)
      {
	params[idx] = m_params[kk].vx_bias0[j];
	idx++;
      }
  
    for(int i = 0; i < m_params[kk].vx_weight2.rows(); i++)
      {
	for(int j = 0; j < m_params[kk].vx_weight2.cols(); j++)
	  {
	    params[idx] = m_params[kk].vx_weight2(i,j);
	    idx++;
	  }
      }
    for(int j = 0; j < m_params[kk].vx_bias2.size(); j++)
      {
	params[idx] = m_params[kk].vx_bias2[j];
	idx++;
      }

    for(int i = 0; i < m_params[kk].vx_weight4.rows(); i++)
      {
	for(int j = 0; j < m_params[kk].vx_weight4.cols(); j++)
	  {
	    params[idx] = m_params[kk].vx_weight4(i,j);
	    idx++;
	  }
      }
    for(int j = 0; j < m_params[kk].vx_bias4.size(); j++)
      {
	params[idx] = m_params[kk].vx_bias4[j];
	idx++;
      }
  }
}



int TireNetwork::load_model(){
  std::cout << "Loading Model\n";
    
  is_loaded = 1;

  for(int kk = 0; kk < num_networks; kk++)
  {
      m_params[kk].weight0 << -2.4255e-01, -7.0588e-02, -2.3864e-02, -1.1146e-01,  9.4910e-04,
      2.3665e-01, -7.6547e-02, -3.0661e-02, -1.6048e-01,  1.3435e-03,
      3.8435e-01,  1.1306e-01, -2.6872e-01,  2.5111e-01,  4.4918e-03,
      6.9476e-02,  6.2066e-01, -5.5215e-02, -6.4403e-02,  8.7934e-04,
      2.9280e-01, -1.2220e-01, -1.2873e-01, -6.8411e-01,  1.4258e-03,
      3.5696e-01, -1.6459e+00, -1.7186e-01, -1.3054e+00,  8.3640e-03,
      2.3000e-01, -6.1460e-01, -3.3222e-02, -1.1617e-01, -1.3987e-03,
      -3.4542e-01, -3.3749e-02, -2.1167e-02, -1.3040e-01, -1.3366e-03;
    m_params[kk].bias0 <<  0.3136, -0.0165, -0.9029,  1.1147, -1.5369, -5.4554, -1.0757, -1.3313;
    m_params[kk].weight2 <<  0.7938, -0.4852, -0.0166, -0.0416, -0.0112, -0.7066, -0.3752,  0.7041,
      0.3218, -1.0782,  1.1807,  1.6408,  0.6843, -1.8775, -0.7722,  0.7403,
      -1.2449,  0.6741,  0.0339,  0.3499,  0.1011,  0.9267,  0.5189, -0.3570,
      -1.0588,  0.1307,  0.0749,  0.5802, -0.4226, -1.6639, -0.3046,  0.7080,
      0.0581,  0.1696, -0.2260,  0.4947,  1.3682,  0.7376, -1.2157, -0.4972,
      -0.1930, -1.0879, -0.0880,  0.0838,  0.5057,  1.1782, -1.4914,  0.2005,
      0.7399, -0.6249,  0.0701,  0.1653, -0.2625,  0.7857,  1.0073, -0.2878,
      -1.3230, -1.1526,  0.8781,  0.6841, -0.7902, -0.2397,  1.2814, -1.7420;
    m_params[kk].bias2 << -0.6658,  0.1108, -0.0645, -0.2862,  0.7468, -0.9752,  0.9822,  1.7467;
    m_params[kk].weight4 << -2.9324, -5.4987,  1.2813, -3.0224,  3.7592, -2.2396,  2.7631,  3.4950,
      -3.6763,  0.1804,  1.9469, -3.6189,  2.9362, -1.9452,  3.8044,  1.5631,
      -3.4620, -0.1360,  2.2266,  1.4514, -0.0413, -1.4595, -1.4343,  1.3484;
    m_params[kk].bias4 << 2.5918, 0.6273, 1.2276;
    
    
    
    m_params[kk].vx_weight0 <<  0.9293,  1.8800,  1.2522, -0.8930;
    m_params[kk].vx_bias0   <<  0.4188,  2.1356,  1.5483,  1.0969;
    m_params[kk].vx_weight2 <<  1.1042,  1.0195,  1.6211,  0.7209, -0.3785, -0.1796, -0.4943, -1.3068, -0.8448, -0.6483, -0.0613,  0.2238, -0.6832, -2.3613, -1.3383,  0.0746;
    m_params[kk].vx_bias2   <<  0.8895, -0.5795,  0.4585,  0.0196;
    m_params[kk].vx_weight4 <<  0.6499, -0.1521,  0.2979,  0.1964;
    m_params[kk].vx_bias4   << -0.0745;
 
  }
  
  out_std << 44.45929437637364, 44.319044796830426, 55.11481922955709;
  in_mean << 0.005049827974289656, 0.5013627409934998, 0.49966344237327576, 0.5000784397125244, 0.05001185089349747;
  in_std_inv << 0.0028585679829120636, 0.2916049659252167, 0.2884257733821869, 0.288765013217926, 0.02888154424726963;
  in_std_inv = in_std_inv.cwiseInverse();
  
  return 0;
}



/*
  no tanh
*/
