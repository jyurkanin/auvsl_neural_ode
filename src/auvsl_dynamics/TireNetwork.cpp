#include "TireNetwork.h"
#include "generated/model_constants.h"
#include <cmath>
#include <cppad/cppad.hpp>
#include <iostream>


Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,TireNetwork::num_in_features> TireNetwork::weight0;
Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,TireNetwork::num_hidden_nodes> TireNetwork::weight2;
Eigen::Matrix<Scalar,TireNetwork::num_out_nodes,TireNetwork::num_hidden_nodes> TireNetwork::weight4;
Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> TireNetwork::bias0;
Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> TireNetwork::bias2;
Eigen::Matrix<Scalar,TireNetwork::num_out_nodes,1> TireNetwork::bias4;
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
  Eigen::Matrix<Scalar,TireNetwork::num_out_nodes,1>    layer4_out;
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
  layer0_out = (weight0*scaled_features) + bias0;
  layer0_out = layer0_out.unaryExpr(&tanh_scalar_wrapper);
  layer2_out = (weight2*layer0_out) + bias2;
  layer2_out = layer2_out.unaryExpr(&tanh_scalar_wrapper);
  layer4_out = (weight4*layer2_out) + bias4;        
  
  // Sign change passivity haxx
  Scalar fx = CppAD::CondExpGt(diff, Scalar(0), layer4_out[0], layer4_out[3]);
  out_vec[0] = relu_wrapper(fx)*CppAD::tanh(1*diff);
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
  
  weight0 <<  8.6618e-02, -4.4287e-01, -3.6348e-02, -3.6585e-02, -1.0653e-01,
    1.3329e-01,  2.0307e-02, -2.4276e-01, -5.1753e-01,  3.4944e-03,
    7.4600e-04,  9.1612e-03, -5.1693e-01,  4.9814e-03,  1.3711e-03,
    1.6810e-03, -1.0607e-02,  6.7861e-01, -7.7098e-02,  3.9396e-02,
    7.6277e-02, -5.7747e-01, -1.5226e-01,  1.1935e+00,  1.2991e-01,
    -8.8606e-01, -1.6855e-01, -4.4853e-01, -9.2104e-02,  4.3377e+00,
    -4.8113e-01,  1.8564e-01;
  bias0 << -0.7269,  1.7868,  1.4498, -0.6849,  1.1558,  0.0634, -2.7032,  7.2892;
  weight2 <<  4.1872e+00, -2.2286e-01,  8.3592e-01, -3.7154e-01, -1.8595e-01,
    -1.8077e+00,  3.6354e+00, -4.3550e-01, -1.6418e-01,  8.9724e-01,
    7.9922e-01,  8.0776e-01, -3.8217e-01,  1.8029e-02, -1.7631e+00,
    -2.5200e+00, -5.6034e-01,  5.5166e-01, -1.4804e+00, -1.5179e+00,
    1.4365e+00,  5.9059e-01, -3.0519e-01,  3.4721e-02,  7.7272e-01,
    1.3312e+00,  9.3108e-01,  9.2185e-01,  1.9445e-01,  6.5389e-02,
    2.2893e-01, -6.1285e-01, -9.9064e-01, -3.4836e-01,  1.1553e-01,
    4.1533e-01,  4.2089e-01,  3.6133e-01, -1.1934e+00,  4.8163e-02,
    2.3472e+00, -1.7792e+00,  5.2336e-01, -6.7307e-01,  1.1635e+00,
    1.3204e+00,  3.0293e+00,  2.8534e+00, -6.5871e-03,  2.0153e-02,
    1.3200e+00,  8.0346e-01,  6.3944e-04, -4.9467e-04, -4.8546e-02,
    8.2892e-03,  1.0414e+00, -1.3534e-01, -7.7075e-01,  2.9723e+00,
    7.7868e-01, -4.7488e-03,  6.8639e-01,  1.0937e+00;
  bias2 << -0.3056,  0.4081,  1.3510,  0.2477, -0.1936, -1.4022, -0.6021, -1.6096;
  weight4 <<  5.9224e+00,  1.5597e+00,  3.9146e+00,  2.6720e+00, -6.5347e+00,
    -3.3334e+00, -4.7684e+00, -2.9885e+00,  4.8018e+00, -3.5219e+00,
    1.9938e+00,  3.5726e+00, -3.4848e+00, -2.0534e+00, -1.9556e+00,
    -3.3136e+00, -1.0818e-02, -3.5562e-02,  5.0956e-03, -1.6256e-01,
    1.1429e-01, -6.7342e-02, -5.1142e+00, -1.4483e+00,  5.9052e+00,
    1.5399e+00,  3.9213e+00,  2.6651e+00, -6.5237e+00, -3.3502e+00,
    -4.7373e+00, -2.9457e+00;
  bias4 << 2.3340, 2.5582, 2.3674, 2.3358;
  out_std << 44.45929437637364, 44.319044796830426, 55.11481922955709;
  in_mean << 0.005049827974289656, 0.5013627409934998, 0.49966344237327576, 0.5000784397125244;
  in_std_inv << 0.0028585679829120636, 0.2916049659252167, 0.2884257733821869, 0.288765013217926;
  in_std_inv = in_std_inv.cwiseInverse();
  
  return 0;
}

