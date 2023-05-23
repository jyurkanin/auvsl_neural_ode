#include "TireNetwork.h"
#include "generated/model_constants.h"
#include <cmath>
#include <cppad/cppad.hpp>
#include <iostream>

TireNetwork::Params TireNetwork::m_params[4];

Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> TireNetwork::out_std;
Eigen::Matrix<Scalar,TireNetwork::num_in_features,1> TireNetwork::in_mean;
Eigen::Matrix<Scalar,TireNetwork::num_in_features,1> TireNetwork::in_std_inv;


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
  
  // Sign change passivity haxx
  out_vec[0] = relu_wrapper(layer4_out[0])*CppAD::tanh(1*diff);
  out_vec[1] = relu_wrapper(layer4_out[1])*CppAD::tanh(-1*in_vec[1]);
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
  assert(params.size() == num_networks*getNumParams());

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
    
  is_loaded = 1;

  for(int kk = 0; kk < num_networks; kk++)
  {
    m_params[kk].weight0 << -4.5866e-02,  9.4588e-01,  8.0724e-02, -3.5639e-01, -2.0311e-03,
      3.2974e-01,  1.7021e-01,  3.1015e-02, -6.8538e-02, -1.8004e-03,
      8.1138e-03,  3.1822e+00, -4.7623e-01,  6.9236e-02,  9.1004e-03,
      -7.3320e-03, -3.1315e-01,  1.0155e-02,  4.9417e-01, -1.7503e-03,
      -1.5960e-01,  2.9414e-01,  1.6486e-01, -5.9909e-02, -4.9959e-03,
      3.4229e-01, -1.0198e-01, -2.7068e-02, -1.3523e-03,  2.2997e-03,
      -8.1350e-01,  1.2769e-02,  2.1557e-02, -2.3651e-02, -1.6901e-03,
      -2.4665e-02,  1.1869e+00, -4.4556e-02,  1.4083e-01,  1.4555e-03;
    m_params[kk].bias0 << -0.2088, -1.1816,  4.4152,  0.0152,  1.5860, -0.7450, -1.0807,  2.5087;
    m_params[kk].weight2 << -1.4027e+00, -8.3032e-01,  5.8206e-01,  5.2517e-01,  1.9189e+00,
      6.7509e-01,  5.3460e-01,  5.3857e-02, -3.9926e-01, -1.4612e+00,
      5.2403e-01, -4.6660e-01, -1.2531e-01, -6.6962e-01,  1.4116e+00,
      -2.3949e+00,  2.2164e-02,  9.2432e-01, -6.7058e-03,  1.2162e-01,
      -3.6446e-01,  6.7866e-01, -6.2339e-01,  1.9524e-01, -3.1702e-02,
      -1.2863e-01,  1.9982e-01, -4.9081e-02,  2.8838e-01,  7.3717e-01,
      -4.1714e-02, -9.1331e-01,  8.8930e-01,  1.4092e-02, -5.3185e-01,
      1.2906e+00,  1.6286e-01, -6.1509e-01,  5.8151e-01,  2.1158e+00,
      -8.7976e-03, -1.3766e+00,  7.2497e-02, -8.7188e-03, -7.0680e-01,
      -9.7039e-01,  2.5589e-01, -2.8993e-01, -6.7451e-02,  9.5124e-01,
      -5.5783e-01, -1.0483e-01,  1.0992e+00,  5.6813e-01, -7.9448e-01,
      -8.7659e-01, -4.1382e+00,  1.1737e+00,  1.3238e+00, -4.5466e+00,
      -3.3171e-01,  2.1471e-01, -6.0234e-01, -7.0394e+00;
    m_params[kk].bias2 <<  0.7765, -1.7752,  0.1596,  1.4282, -0.1220, -0.9163, -0.3326, -0.2775;
    m_params[kk].weight4 <<  3.7718e+00, -2.5548e+00, -3.9803e+00,  4.1168e+00, -3.2620e+00,
      -2.0412e+00,  4.8447e+00,  4.6855e+00,  2.7197e+00, -2.3709e+00,
      -9.0612e-02,  2.9905e+00, -3.6150e+00, -1.0228e+00,  1.3027e-01,
      4.1716e+00,  1.3025e+00, -2.7016e-01,  2.9979e+00,  1.8089e+00,
      -2.3960e-03, -2.6877e+00,  5.6502e-02,  7.3332e-02;
    m_params[kk].bias4 << 2.1050, 2.2356, 1.6341;

  }
  
  out_std << 44.45929437637364, 44.319044796830426, 55.11481922955709;
  in_mean << 0.005049827974289656, 0.5013627409934998, 0.49966344237327576, 0.5000784397125244, 0.05001185089349747;
  in_std_inv << 0.0028585679829120636, 0.2916049659252167, 0.2884257733821869, 0.288765013217926, 0.02888154424726963;
  in_std_inv = in_std_inv.cwiseInverse();
  
  return 0;
}



/*
  no tanh
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
*/
