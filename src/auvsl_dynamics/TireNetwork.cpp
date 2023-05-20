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
  Scalar slip_lon = CppAD::abs(diff);
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
  return 1;
  
  weight0 << -1.7484e-01,  1.6525e+00, -6.8996e-02, -5.9168e-01, -3.4555e-01,
    4.6433e-01,  3.1354e-01,  1.9098e-01,  1.1104e-01, -5.4889e-01,
    -2.5016e-03, -1.2739e-01, -7.0772e-01, -1.9510e-01,  1.2977e-02,
    -1.2896e-01, -3.5542e-01,  7.6961e-02,  2.8606e-02, -2.9580e-01,
    2.7108e-01, -1.9145e+00,  8.5818e-02, -2.5109e-01, -5.1642e-01,
    1.8547e-01,  1.9089e-01,  3.8624e-01, -2.5881e-01, -2.4439e-03,
    4.8472e-04,  1.7329e-03;
  bias0 <<  2.0486,  1.5814, -0.8503, -2.6291,  0.8921, -3.7739,  1.4840,  0.1440;
  weight2 << -0.4313,  1.0166,  0.9065, -0.8379,  0.6155,  3.0698,  1.1143, -0.6207,
    0.0440, -0.0322,  0.0723, -0.4120, -0.0300,  0.0468,  0.0812,  3.0522,
    -0.3356, -0.8850,  1.0218, -1.3081, -0.2968, -0.4051,  0.9048,  0.0117,
    0.8444,  1.0906,  2.0426,  1.9503, -1.0902,  0.3945,  1.4630,  1.1683,
    -2.0804,  1.5648, -1.0079, -0.0329, -0.9311, -2.5838, -1.1102,  1.0741,
    0.0104,  0.0073,  0.0236, -0.0033,  0.0148, -0.0742,  0.0072,  1.9988,
    -0.2176, -0.2047,  0.6015, -0.9912, -0.4051,  0.7649, -0.1259, -0.0506,
    -0.2627, -0.8253,  0.0100,  0.3662, -0.3856, -1.4246,  0.4105,  0.1293;
  bias2 << -0.3422,  0.4901,  0.9934, -0.7216,  0.4807, -0.7637,  0.5814, -0.6752;
  weight4 <<  0.3545, -0.1442,  2.4540, -3.1774, -5.8501, -0.7761,  4.0456, -5.6217,
    -2.1220, -0.1018,  2.2535, -2.1167, -5.7603, -0.8064,  3.9032, -3.4517,
    -0.0415, -2.2064,  0.8490, -0.0772,  0.0278, -3.6400, -0.0999, -0.0941;
  bias4 << 1.3820, 1.1332, 2.5680;
  out_std << 44.45929437637364, 44.319044796830426, 55.11481922955709;
  in_mean << 0.005049827974289656, 0.5013627409934998, 0.49966344237327576, 0.5000784397125244;
  in_std_inv << 0.0028585679829120636, 0.2916049659252167, 0.2884257733821869, 0.288765013217926;
  in_std_inv = in_std_inv.cwiseInverse();
  
  return 0;
}

