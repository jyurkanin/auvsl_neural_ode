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
  Scalar slip_lon = diff;
  Scalar slip_lat = in_vec[1];
  Scalar tire_abs = in_vec[2];
  
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
  // out_vec[0] = relu_wrapper(layer4_out[0])*CppAD::tanh(1*diff);
  // out_vec[1] = relu_wrapper(layer4_out[1])*CppAD::tanh(-1*in_vec[1]);
  // out_vec[2] = relu_wrapper(layer4_out[2])/(1 + CppAD::exp(-1*in_vec[3]));

  out_vec[0] = layer4_out[0];
  out_vec[1] = layer4_out[1];
  out_vec[2] = layer4_out[2];
  
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
  
  weight0 <<  3.0374e-01,  9.7640e-01, -1.2625e-01, -4.0828e-01, -1.6689e-02,
    2.7146e+00,  8.9340e-02, -4.8581e+00, -2.2156e-01,  2.7260e-01,
    -4.0986e-02,  4.1852e-01,  4.6267e-01,  2.3403e-01, -3.7966e-02,
    -8.8981e-01, -3.1597e-01, -1.9859e-01,  3.5771e-02, -4.8057e-01,
    1.6139e-01, -1.9877e-01,  3.6707e-02,  7.6257e-01,  3.4773e-01,
    4.1630e-01, -6.2941e-02,  4.2125e-01, -4.6820e-01, -1.1292e-01,
    1.4763e-02, -2.0136e-01, -9.7823e-03,  1.0786e+01,  1.7691e+00,
    1.9786e-01, -1.2142e-01,  3.1106e+00, -2.5012e-02,  1.5760e-01,
    4.4736e-02,  1.8055e+00,  9.0331e-02,  5.1150e+00, -2.8198e-01,
    4.1979e-01, -5.1738e-02,  2.6085e-01,  7.9682e-02,  1.9495e+01,
    1.2919e+00, -6.9027e-02, -4.3577e-01,  4.7756e-01, -5.8692e-02,
    -1.8856e-01, -3.2834e-01, -5.0346e-02,  8.6384e-03, -6.9466e-02,
    2.1210e-01,  1.0568e-01, -3.4775e-03, -3.5508e-01;
  bias0 << -0.5242, -0.0108, -0.7373,  1.3806, -1.2357, -0.3286, -0.0602,  1.3429,
    0.0288,  0.1904, -0.0323,  0.6060, -0.1077, -1.1632, -0.6947, -0.1633;
  weight2 << -4.5732e-01, -3.8300e-01,  2.1040e-01,  5.8647e-01, -3.8317e-01,
    2.2049e-01,  1.5469e-02,  1.8198e-01,  6.5511e-01, -7.2024e-01,
    -7.4954e-01, -2.5409e-01, -7.6570e-01,  1.2381e-01, -4.5078e-01,
    3.4291e-01, -4.8460e-01,  7.6178e-01,  6.9149e-02, -4.2635e-01,
    -1.0428e-02, -7.8175e-01, -4.1582e-01,  1.0025e+00,  1.1273e+00,
    -8.5713e-01, -5.4743e-02, -9.0249e-01, -9.4516e-01, -8.5908e-02,
    -2.3230e-01,  1.4823e+00, -6.5750e-01, -2.6642e-01, -2.3571e-01,
    3.3039e-01, -1.7146e-01,  4.9808e-01,  2.4269e-01,  3.4697e-01,
    1.0653e-01, -2.0261e-01, -2.8711e-01, -3.1399e-01, -3.3266e-01,
    1.1129e-01, -1.9580e-01,  3.6652e-01, -4.2642e-01,  5.3730e-01,
    1.3868e-01, -2.2352e-01,  5.6761e-01, -3.3963e-01, -2.1482e-01,
    -1.3426e-01,  6.7615e-01, -5.8513e-01, -6.7598e-01,  2.6946e-01,
    -5.7988e-01,  5.9233e-02,  2.5304e-01,  1.7607e-01,  3.9958e-01,
    4.8281e-02,  4.0452e-01, -1.1319e-01,  1.2755e-01,  2.3508e-01,
    4.6343e-01,  2.8898e-01, -4.0543e-01, -9.8739e-02, -2.0292e-01,
    -6.4234e-01,  3.9551e-01, -2.5390e-01,  7.9074e-01,  1.9190e-01,
    5.0755e-01,  5.5323e-01, -1.5489e-01,  5.4041e-02, -7.9860e-02,
    1.8665e-01, -8.7678e-02,  1.6216e-01, -6.3749e-01,  4.6978e-01,
    3.9568e-02,  2.4194e-02,  7.3916e-01, -5.0547e-02, -5.7392e-01,
    -2.4480e-01,  1.0197e-01, -1.1359e-01, -2.5692e-02, -5.5258e-02,
    -8.7553e-01,  3.5494e-01, -9.9054e-01,  1.9665e-01,  3.7975e-01,
    -6.2119e-01, -7.0158e-01,  3.1935e-01, -4.3702e-01,  7.3563e-01,
    -9.2760e-01, -5.3859e-01, -1.0111e-01,  5.2241e-01,  8.9742e-01,
    1.1406e+00, -1.5009e-01, -1.2456e+00,  5.1395e-01, -2.2980e-01,
    4.5429e+00,  2.2383e+00,  2.7559e+00,  1.2775e+00,  4.0646e+00,
    1.4823e+00,  2.2412e-01,  1.1985e+00, -7.5983e-03, -5.5404e-03,
    1.8912e-01, -2.8110e-02, -3.5815e-02, -5.6346e-02, -7.0309e-02,
    5.8725e-01,  5.5179e-03,  1.2379e-02,  1.0848e-02,  9.4581e-02,
    -5.1215e-03,  6.4388e-02,  1.0103e+00, -8.2127e-02,  1.4215e-02,
    -5.6967e-03, -1.2367e-01, -1.3558e-01, -6.0225e-01, -1.6774e-01,
    8.8964e-02, -9.7503e-01, -1.5135e-01,  1.0041e-01,  3.2757e-03,
    -8.5217e-01,  9.8559e-02,  2.0723e-01, -3.2308e-01,  6.1934e-01,
    3.0698e-01,  2.2811e-02,  6.1753e-02,  1.0906e-01,  4.2759e-01,
    7.6962e-01,  1.9896e-01, -6.5312e-01, -3.6926e-01,  4.4868e-03,
    2.4015e-01, -1.5826e-01,  5.1071e-01, -5.8943e-02, -1.6079e-01,
    -5.3493e-02, -5.5053e-01, -6.9785e-01, -5.0944e-02,  1.0411e-02,
    -3.4357e-02, -5.9274e-02,  2.4201e-01, -6.1761e-01,  6.0207e-01,
    -4.4100e-01, -5.3539e-02, -1.6799e-01, -6.4943e-01, -6.8586e-02,
    -2.5855e-01,  7.8537e-01, -3.7740e-01, -4.2766e-01,  6.4542e-01,
    -3.1216e-01, -3.0405e-01, -8.6520e-01, -3.3411e-01,  7.8369e-01,
    7.8949e-01, -7.5720e-01, -8.4985e-01, -1.8693e+00, -8.5330e-01,
    4.5245e-01, -2.5383e-01, -2.4103e-01, -1.4727e+00, -8.0742e-01,
    -8.7079e-02,  6.2936e-01,  9.1995e-02,  4.5808e-01,  5.8049e-01,
    -5.1474e-01,  1.1399e+00, -7.4987e-01, -5.4609e-02, -2.4618e-01,
    -1.2299e+00,  4.8379e-01, -1.7374e-01,  5.0071e-01, -2.0415e-01,
    -4.2310e-01,  6.5879e-01,  2.6857e-01,  1.5271e-01,  3.1179e-01,
    -3.1691e-01, -1.2946e-01, -6.6477e-01,  5.9847e-01,  5.1950e-01,
    3.7873e-01,  6.5638e-01, -3.6823e-02,  4.3031e-01, -2.6659e-01,
    -6.4411e-01, -2.9667e-01, -5.5129e-01,  9.6086e-01,  4.8752e-01,
    1.1682e+00,  3.3650e-02, -2.6886e-01,  9.7405e-02, -1.6832e-01,
    4.4771e-01, -1.5595e-02, -3.1031e-02,  9.2683e-01, -9.1078e-02,
    -8.6340e-01;
  bias2 <<  0.3950, -0.3476, -0.9067, -0.4917, -1.0814,  0.4465,  0.5778,  0.2693,
    0.2646, -0.4167, -0.3626, -1.3232,  0.8156, -0.2877, -0.6579,  0.1667;
  weight4 <<  1.0504,  0.0226, -0.4913, -0.0581,  0.0035, -1.6302, -1.0078,  0.0065,
    -0.1680, -0.0204,  0.2901, -1.2588, -1.0152, -0.7699, -0.2776, -0.1286,
    -0.1483,  0.7173,  1.1586, -0.6823, -0.3189, -1.1388,  0.0573,  0.2402,
    -0.3024,  0.6644, -0.7866, -0.9359,  0.4102, -0.8495,  1.0245, -1.2078,
    -0.0244, -0.0145, -0.0675,  0.0222, -1.2528,  0.0055,  0.0202, -0.0041,
    -2.5188,  0.6904, -0.0078,  0.0506,  0.0101,  0.0183,  0.0033,  0.0118;
  bias4 <<  0.1176, -0.3203,  1.1335;
  out_std << 44.45929437637364, 44.319044796830426, 55.11481922955709;
  in_mean << 0.005049827974289656, -1.6948217307799496e-05, 0.00029757287120446563, -0.00045066422899253666;
  in_std_inv << 0.0028585679829120636, 0.5799984931945801, 0.576934278011322, 0.5774632096290588;
  in_std_inv = in_std_inv.cwiseInverse();
  
  return 0;
}

