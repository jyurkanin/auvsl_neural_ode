#include "TireNetwork.h"
#include "generated/model_constants.h"
#include <cmath>
#include <cppad/cppad.hpp>
#include <iostream>


Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,TireNetwork::num_in_features> TireNetwork::weight0;
Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,TireNetwork::num_hidden_nodes> TireNetwork::weight2;
Eigen::Matrix<Scalar,TireNetwork::num_out_features,TireNetwork::num_hidden_nodes> TireNetwork::weight4;
Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> TireNetwork::bias0;
Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> TireNetwork::bias2;
Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> TireNetwork::bias4;
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


// vx vy w zr kc kphi n0 n1 phi
void TireNetwork::forward(const Eigen::Matrix<Scalar,9,1> &in_vec, Eigen::Matrix<Scalar,num_out_features,1> &out_vec){
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> layer0_out;
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> layer2_out;
  Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> layer4_out;
  Eigen::Matrix<Scalar,TireNetwork::num_in_features,1> scaled_features;
  Eigen::Matrix<Scalar,TireNetwork::num_in_features,1> bekker_vec;
  
  // Changes features to cross the origin
  Scalar tire_tangent_vel = in_vec[2] * Jackal::rcg::tire_radius;
  Scalar diff = tire_tangent_vel - in_vec[0];
  Scalar slip_lon = CppAD::abs(diff) / (CppAD::abs(tire_tangent_vel) + 1e-6);
  Scalar slip_lat = CppAD::abs(in_vec[1]);
  Scalar tire_abs = CppAD::abs(in_vec[2]);
  
  bekker_vec[0] = in_vec[3];
  bekker_vec[1] = slip_lon;
  bekker_vec[2] = tire_abs;
  bekker_vec[3] = slip_lat;
  // bekker_vec[4] = in_vec[4];
  // bekker_vec[5] = in_vec[5];
  // bekker_vec[6] = in_vec[6];
  // bekker_vec[7] = in_vec[7];
  // bekker_vec[8] = in_vec[8];

  // Apply scaling after calculating the bekker features from kinematics
  scaled_features = (bekker_vec - in_mean).cwiseProduct(in_std_inv);

  // Actual NN math
  layer0_out = (weight0*scaled_features) + bias0;
  layer0_out = layer0_out.unaryExpr(&tanh_scalar_wrapper);
  layer2_out = (weight2*layer0_out) + bias2;
  layer2_out = layer2_out.unaryExpr(&tanh_scalar_wrapper);
  layer4_out = (weight4*layer2_out) + bias4;        
  
  // Sign change passivity haxx
  out_vec[0] = relu_wrapper(layer4_out[0])*CppAD::tanh(1*diff);
  out_vec[1] = relu_wrapper(layer4_out[1])*CppAD::tanh(-1*in_vec[1]);
  out_vec[2] = relu_wrapper(layer4_out[2])/(1 + CppAD::exp(-1*in_vec[3]));
  
  // Scale output
  out_vec = out_vec.cwiseProduct(out_std);
}


int TireNetwork::getNumParams()
{
  return (weight0.size() +
	  bias0.size() +
	  weight2.size() +
	  bias2.size() +
	  weight4.size() +
	  bias4.size());
}

void TireNetwork::setParams(const VectorS &params, int idx)
{
  assert(params.size() == getNumParams());
  for(int i = 0; i < weight0.rows(); i++)
  {
    for(int j = 0; j < weight0.cols(); j++)
    {
      weight0(i,j) = params[idx];
      idx++;
    }
  }
  for(int j = 0; j < bias0.size(); j++)
  {
    bias0[j] = params[idx];
    idx++;
  }
  
  for(int i = 0; i < weight2.rows(); i++)
  {
    for(int j = 0; j < weight2.cols(); j++)
    {
      weight2(i,j) = params[idx];
      idx++;
    }
  }
  for(int j = 0; j < bias2.size(); j++)
  {
    bias2[j] = params[idx];
    idx++;
  }

  for(int i = 0; i < weight4.rows(); i++)
  {
    for(int j = 0; j < weight4.cols(); j++)
    {
      weight4(i,j) = params[idx];
      idx++;
    }
  }
  for(int j = 0; j < bias4.size(); j++)
  {
    bias4[j] = params[idx];
    idx++;
  }
}

void TireNetwork::getParams(VectorS &params, int idx)
{
  assert(params.size() == getNumParams());
  
  for(int i = 0; i < weight0.rows(); i++)
  {
    for(int j = 0; j < weight0.cols(); j++)
    {
      params[idx] = weight0(i,j);
      idx++;
    }
  }
  for(int j = 0; j < bias0.size(); j++)
  {
    params[idx] = bias0[j];
    idx++;
  }
  
  for(int i = 0; i < weight2.rows(); i++)
  {
    for(int j = 0; j < weight2.cols(); j++)
    {
      params[idx] = weight2(i,j);
      idx++;
    }
  }
  for(int j = 0; j < bias2.size(); j++)
  {
    params[idx] = bias2[j];
    idx++;
  }

  for(int i = 0; i < weight4.rows(); i++)
  {
    for(int j = 0; j < weight4.cols(); j++)
    {
      params[idx] = weight4(i,j);
      idx++;
    }
  }
  for(int j = 0; j < bias4.size(); j++)
  {
    params[idx] = bias4[j];
    idx++;
  }
}



int TireNetwork::load_model(){
  std::cout << "Loading Model\n";
    
  is_loaded = 1;
  weight0 << -8.0111e-02,  4.1367e-01,  1.7542e+00,  2.1826e-01,  5.8170e-02,
    -1.5887e+01,  1.0438e-01, -2.0220e-01,  3.7334e-02,  1.8921e-01,
    -2.1129e-01,  3.2440e-01,  8.5260e-02, -2.4545e-01,  3.3552e-01,
    -5.3833e-01,  1.0188e-01, -1.6125e-01, -6.6306e-01, -3.9503e-02,
    8.5042e-01,  1.0519e-02, -7.3232e-03,  1.1329e-02,  3.1250e-01,
    -4.2915e-03,  8.0248e-03, -1.8129e-02, -1.1126e-01, -9.1475e-01,
    -4.2029e-01,  6.8803e-01;
  bias0 <<  3.6161, -0.9196,  0.8711, -0.1713, -0.7475,  1.2725, -0.6405,  1.2808;
  weight2 <<  -0.1777,  -2.3559,   1.6973,   0.2749,  -1.1225,  -0.8640,  -0.3778,
    1.5926,   2.0911,  -4.7534,   0.1632,   3.0805,  -1.1520,  -1.5457,
    1.0019,  -0.8370,   0.4756,  -0.8038,   0.4678,   0.4870,  -0.1687,
    1.0315,   0.4125,  -0.0549,   0.8001,  -1.3324,  -0.9877,   1.0525,
    -0.4101,   0.3463,   0.2381,  -0.6970,   0.5864,  -0.4744,  -0.3472,
    -0.9928,   0.0965,  -0.4008,   0.1103,   0.7823,  -0.2550,   0.3226,
    0.3143,   0.0316,   0.0443,  -0.7172,  -1.4976,  -0.0338,   1.7305,
    -16.2732,  -1.3519,   0.0249,  -0.7740,  -0.6086,   1.8489,  -3.6355,
    1.5509,  -2.0025,  -1.2693,   0.5963,  -0.2847,  -1.1850,  -0.7573,
    0.4362;
  bias2 <<  1.4955,  0.7743,  0.3593, -0.7974, -0.6725,  0.4339, -0.6766, -0.6389;
  weight4 <<  3.6329,  2.9496,  3.1018, -3.4174, -3.5513,  0.3270, -9.0172, -2.1828,
    1.6583,  2.2588,  2.6772, -2.9587, -3.0235,  0.8244, -5.1252, -3.3849,
    0.7559, -0.1832,  1.5231, -0.4264, -0.4402, -4.8023,  0.0357, -0.4182;
  bias4 << 2.2084, 1.9601, 3.0297;
  out_std << 44.45929437637364, 44.319044796830426, 55.11481922955709;
  in_mean << 0.005049827974289656, 59.35271453857422, 0.49966344237327576, 0.5000784397125244;
  in_std_inv << 0.0028585679829120636, 1799.4019775390625, 0.2884257733821869, 0.288765013217926;
  in_std_inv = in_std_inv.cwiseInverse();
  
  return 0;
}

