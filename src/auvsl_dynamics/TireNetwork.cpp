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
  bekker_vec[4] = in_vec[4];
  bekker_vec[5] = in_vec[5];
  bekker_vec[6] = in_vec[6];
  bekker_vec[7] = in_vec[7];
  bekker_vec[8] = in_vec[8];

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
  
  weight0 <<  1.8180e-01,  1.4623e-02, -4.7478e-01, -1.6304e-02,  8.6695e-02,
         1.6159e-01, -7.7599e-01, -1.4306e-02,  7.4051e-02,  8.4975e-02,
        -5.1739e-01, -3.5777e-02,  3.3356e-03,  2.2787e-02,  2.5119e-02,
        -1.7598e-01,  7.5733e-01,  4.5451e-02, -6.4119e-03, -4.9849e-01,
         3.5988e-01,  5.0708e-03, -7.1760e-03, -1.7082e-02,  2.6247e-02,
         1.5712e+00, -1.1000e-02, -9.4163e-03, -7.0207e-01, -3.9780e-02,
         1.6321e+00,  4.8687e-02,  9.2708e-02, -1.9440e-01, -3.8851e-02,
        -1.1088e-01, -1.9499e-01,  1.5972e-01,  8.2362e-02,  2.0585e-01,
        -1.0071e-02, -1.5088e-02, -2.3660e-01,  3.5219e-02, -2.4608e-01,
         7.2219e-02,  2.7737e-01, -1.7982e-01, -1.9645e-03,  6.2135e-02,
         1.2827e-01, -7.1499e-01,  5.1068e-01,  3.6365e-02, -6.1541e-01,
        -9.7430e-02,  1.3249e-01,  4.3042e-03,  5.0672e-03,  5.7213e-03,
         5.6355e-01,  4.2069e-03,  7.2305e-02, -7.4756e-02,  9.7104e-01,
         1.1591e-01,  5.3887e-01, -3.0682e-02, -6.0822e-02,  5.3242e-02,
         5.9241e-02, -7.8642e-02,  1.3762e-02, -1.7779e+00, -4.5741e-01,
        -2.2563e-02, -2.9385e-03, -4.5430e-03, -4.0485e-02,  5.1288e-01,
         4.6985e-02,  1.6627e-02,  3.1379e-01, -1.2868e+00, -2.1516e-02,
         3.0056e-02,  5.0716e-02, -2.2094e-01,  2.9953e-01, -3.2340e-02,
         3.4821e-02, -1.5713e-01,  1.5327e-01, -2.2679e-02, -2.7795e-01,
        -5.0646e-01,  1.0269e-01, -5.0466e-02,  1.4097e-02,  1.6669e-02,
         6.1891e-01,  6.0700e-02, -1.3948e-02, -8.7852e-04,  6.7574e-03,
         1.4741e-01,  4.7704e-01, -5.1809e-02,  1.8729e-01, -1.2556e-01,
        -4.5475e-02, -5.2703e-02, -6.9853e-03,  4.0129e-03, -9.1021e-02,
         2.8629e-02, -1.2879e-01,  1.1130e-01, -6.0996e-01,  7.8486e-01,
        -3.5163e-03,  2.3019e-02,  4.0078e-02, -3.0681e-01, -5.9801e-01,
         8.0310e-02,  1.9981e-01, -3.2261e-01,  4.0017e-01, -1.3394e-02,
         5.4306e-02,  1.0081e-01, -6.8772e-01, -2.8458e-01,  1.0887e-01,
        -4.5658e-01, -2.1181e-02,  8.2971e-03, -1.2510e-02,  3.3144e-03,
         6.6302e-03, -1.4003e-01, -2.9014e-02,  3.0554e-02,  7.4625e-02,
        -6.3923e-01, -2.1692e-01,  5.7664e-01, -2.1532e-02, -2.5493e-02,
        -1.7887e-01, -2.1092e-01,  1.0802e-02, -8.9590e-02, -1.0843e+00,
        -5.0757e-02, -3.6909e-02, -6.6638e-03, -1.8999e-02,  2.7256e-02,
         5.3140e-01, -7.6699e-02, -1.0919e-01, -1.2620e+00,  3.8873e-01,
        -7.1206e-02, -1.7786e-02, -1.7876e-02,  2.8967e-01,  1.7647e-01,
        -7.0304e-03,  3.2932e-02,  6.6793e-01, -5.1157e-01, -3.3286e-02,
        -2.0192e-02, -3.5162e-02,  2.3662e-02,  5.7904e-01,  1.1952e-01;
bias0 << -0.7198,  0.8125,  3.5000,  0.1204,  0.0338,  1.5167, -0.8529,  2.7609,
        -3.9390, -1.8917, -1.4577,  0.6428, -1.2852, -0.3870, -1.6843, -1.0315,
        -1.5955, -1.8812, -0.8086,  0.0404;
weight2 << -6.7564e-02, -9.1913e-02,  1.3641e-02,  1.7886e-01,  1.4111e-01,
         5.9752e-02,  1.2150e-01,  1.4247e+00,  1.2156e-01,  1.9175e-02,
         1.5773e-01,  1.3860e-01, -1.6544e-01,  7.2051e-02, -2.1382e-01,
         4.9244e-01,  9.8075e-01, -2.8266e-01, -1.4167e-01,  1.5407e-01,
         1.9390e-01,  1.3674e+00,  2.0296e+00,  1.4504e-01, -7.7380e-01,
        -7.5891e-01,  8.9825e-01, -1.1273e+00, -1.8420e+00,  4.6870e+00,
         2.3758e-01,  8.9559e-01,  6.5942e-01, -5.8167e-01, -7.4813e-01,
         6.0383e-01,  8.8024e-01, -1.2115e+00, -1.1329e+00, -5.6207e-01,
        -1.0715e+00,  4.6552e-01,  1.9154e+00, -3.7463e-01,  9.3714e-01,
        -3.7074e-01,  5.8777e-01,  2.0214e+00, -9.3190e-01,  7.0327e-01,
         8.7517e-01,  6.1598e-01, -6.0824e-02, -3.2201e-01, -3.9547e-01,
         5.4534e-01, -5.8143e-01, -8.1861e-01, -3.8026e+00,  1.1951e+00,
         8.2596e-01, -9.0792e-01, -2.6830e+00, -2.9355e-01, -7.3887e-01,
         1.4613e+00, -8.0090e-01, -1.0736e+00,  2.3171e+00, -8.6921e-01,
        -3.8883e-01, -9.0473e-01, -1.1265e-01,  1.1550e+00,  1.3337e+00,
        -5.2110e-01, -1.3867e-01,  1.0120e+00,  2.3832e-01, -1.1034e+00,
         6.7951e-02,  1.5358e-01, -1.3607e-01, -1.0573e-01, -1.4211e-01,
        -1.8427e-02, -7.2858e-02, -9.4750e-01,  8.7121e-01,  1.9236e-01,
        -4.7776e-01,  2.9489e-01, -4.5617e-02,  1.4612e-01,  5.1064e-01,
        -4.0241e-01,  1.6252e-01, -4.7284e-01, -9.0385e-01,  4.6720e-01,
        -3.7116e-02, -8.0074e-02, -1.8193e-02,  2.4116e-03,  2.0776e-01,
         4.8528e-02,  1.3868e-01,  1.0409e+00,  7.5291e-02, -1.0545e-01,
         1.7529e-01,  1.6956e-01, -9.1536e-01,  1.1586e-02, -1.7890e-01,
         6.2039e-01,  1.0647e-01,  2.4526e-02,  3.3738e-02, -4.2164e-03,
         4.9310e-01, -3.2625e-02, -3.1578e-02,  4.9116e-02, -7.3669e-01,
        -1.3226e-01, -3.1603e-01,  1.9755e+00,  5.3293e-01, -2.9535e-01,
         1.0777e-02,  3.0496e-01,  2.8603e-01,  5.6270e-01, -7.1038e-01,
        -2.0073e+00,  8.2493e-01, -2.8095e-01, -1.8654e-02,  2.7714e-01,
        -6.9746e-01,  1.0982e+00,  1.8848e+00,  6.9224e-02, -6.1089e-01,
        -5.6640e-01,  8.8520e-01, -5.9643e-01, -1.7227e+00,  1.4011e+00,
         8.2830e-01,  8.4510e-01, -2.0177e-01, -7.0153e-01, -6.2103e-01,
        -9.3240e-01,  1.2099e-01, -7.7301e-01, -4.6801e-01,  1.0282e+00,
         1.0482e+00, -1.1242e-01, -6.3352e-01,  1.1806e+00, -6.2976e-01,
        -1.9772e-02, -3.7915e-01, -9.8244e-01,  8.0767e-01,  9.5708e-01,
        -5.1176e-01, -6.5612e-01,  1.3623e+00,  1.0160e+00,  3.4498e-01,
         4.1766e-01,  2.6828e-01,  2.9215e-01, -1.8727e-01, -4.9003e-01,
         7.6502e-01, -4.8078e-01, -2.6339e+00, -1.1673e+00, -2.0159e+00,
         1.0654e+00, -7.7778e-01, -1.2894e+00,  2.0600e+00, -9.6011e-01,
        -3.7322e-01, -1.1079e+00,  1.7079e-01,  9.8040e-01,  1.7747e+00,
        -3.7112e-01,  1.4010e-01,  1.7105e+00, -1.3180e-01, -1.0382e+00,
         7.0460e-01,  1.2413e-01,  1.5168e-03, -1.4684e-01,  9.9842e-02,
        -2.4449e-01,  1.5419e+00, -1.5125e+00, -3.0718e-01, -5.2130e-02,
        -6.5519e-02, -6.9981e-04, -1.3374e+00, -1.8289e-01,  8.1674e-01,
         2.1787e+00, -1.5724e-01,  2.2797e-01,  8.0102e-02, -9.7073e-02,
        -5.5669e-01, -1.4991e-01,  2.1848e+00,  3.2374e+00,  6.9881e-01,
         3.5112e-01,  5.6130e-01,  4.4007e+00, -2.2259e+00,  1.3598e+00,
         4.2739e-01,  1.2512e+00, -3.6680e-01, -4.8930e-04, -1.3658e+00,
         8.1705e-01,  4.0921e-01, -2.7972e+00,  5.8762e-01,  2.7931e+00,
        -5.3059e-01,  1.1182e+00,  2.0770e+00, -5.1508e-03,  9.3641e-02,
        -9.5022e-01,  8.4485e-01, -1.9870e-01, -1.6052e+00,  1.8697e+00,
         7.9673e-01,  8.5647e-01, -5.9544e-01, -8.4827e-01, -1.0392e+00,
         2.8369e-01, -4.2685e-02, -7.7257e-01, -4.5354e-01,  6.8771e-01,
         8.5428e-01, -1.0911e+00, -9.7355e-01,  5.0134e-01,  3.0657e-01,
         4.6777e-01, -9.6248e-01,  1.9639e-01,  1.0975e+00, -4.1883e-01,
        -6.0677e-01, -4.0293e-01,  1.3315e+00,  5.7658e-01,  5.4249e-01,
        -4.3756e-02, -1.8373e-01,  5.1010e-01,  1.1114e+00, -1.2927e+00,
        -4.3463e-01,  6.5396e-01,  1.5580e+00,  2.4652e-02, -2.9384e-01,
        -5.5641e-01,  6.5493e-01, -8.4132e-01, -1.4724e+00,  5.8286e-01,
         4.1586e-01,  3.1526e-01, -1.3583e+00, -6.3574e-01, -5.4998e-01,
         9.1527e-01, -1.7576e-02, -2.5134e-01, -1.2265e-01,  6.8497e-01,
         4.6993e-01, -6.0897e-01, -1.6662e+00, -4.8529e-02,  3.4453e-01,
         8.2688e-01,  4.9569e-01, -1.6491e-01,  1.3137e+00, -1.0163e+00,
        -8.7477e-01, -7.4506e-01, -4.8144e-02,  8.5202e-01,  1.8853e-01,
         2.7293e+00, -1.5792e-01,  6.4704e-01,  2.6936e-01, -4.2037e-01,
         1.2490e+00, -1.2533e+00, -2.9482e+00, -1.5580e-02, -3.8735e-01,
         9.4839e-01, -8.0245e-01, -2.3778e-01,  2.4471e+00, -1.4299e+00,
        -1.0347e+00, -1.0214e+00, -1.3459e-01,  1.3762e+00,  1.2492e+00,
        -3.6224e-01, -2.1796e-01,  1.2548e+00,  5.4924e-01, -1.0791e+00,
         2.7278e-01, -2.0053e-01, -2.1316e-01, -4.4867e-01, -4.7980e-01,
        -1.1344e-01, -1.9121e-01, -3.1205e-02,  6.7967e-01,  2.1390e-02,
        -1.1344e-01, -5.1857e-01,  1.4710e+00, -5.6180e-03,  8.2956e-01,
         5.5298e-02,  2.4327e-01, -4.7246e-02, -1.6207e+00, -1.8503e-01,
        -1.1012e+00,  1.9080e+00,  3.8662e+00,  6.3015e-02, -5.3026e-01,
        -2.2069e+00,  1.4803e+00, -1.9219e-01, -3.3503e+00,  1.3713e+00,
         1.5800e+00,  1.5196e+00, -4.6590e-01, -1.4718e+00, -1.7506e+00,
         4.6626e-01,  5.9085e-02, -1.4687e+00, -6.5753e-01,  1.5534e+00,
        -7.5942e-01,  1.2058e+00,  2.8958e+00, -8.7944e-03, -5.9459e-01,
        -6.4741e-01,  1.1006e+00, -1.3468e+00, -6.6844e-01,  1.5451e+00,
         4.0061e-01,  1.2303e+00, -1.1165e+00, -9.9790e-01, -1.3953e+00,
         2.3892e-01, -1.6521e-01, -1.3192e+00, -2.8524e-03,  8.7682e-01;
bias2 <<  0.7567, -1.0695,  1.2340,  0.2581, -1.6291, -0.6923,  0.3082,  0.8092,
        -0.8211, -0.5793, -0.3339,  0.7459, -0.5175, -1.2051,  0.7070, -0.7540,
         0.5351, -1.1120, -0.6009,  0.9711;
weight4 << -4.2018, -0.0253, -6.3545,  2.9483, -2.1081, -1.3250,  3.5704,  2.2603,
        -3.1544,  3.0909, -3.2304, -4.7510, -1.9611, -1.1903,  2.9457, -3.3833,
         3.0771, -3.2429, -3.5679,  3.9633, -1.5002, -0.0271,  0.2828,  3.7078,
        -2.2501, -1.5503,  1.7750,  2.3042, -2.5008,  4.0497, -1.7662, -4.9976,
        -1.2197, -2.0602,  2.2194, -2.0030,  2.3902, -2.2784, -1.6849,  0.0457,
         0.0511, -3.0932, -0.0895,  0.3921, -0.3921, -0.0890,  0.0065,  2.9164,
        -0.0226,  0.1837, -0.1636,  0.0297, -1.3676,  0.5281, -3.2500, -2.2671,
        -0.2548,  0.0568, -3.1768,  0.2244;
bias4 <<  2.5221,  1.7490, -0.2870;
out_std << 29.010036340552517, 37.073029267252366, 78.16212403454449;
in_mean << 0.005049172788858414, 0.5014784932136536, 0.5003626942634583, 0.500156581401825, 60.011775970458984, 1998.331298828125, 0.7998817563056946, 0.10001819580793381, 0.34496328234672546;
in_std_inv << 0.0028567721601575613, 0.2911893427371979, 0.2885623872280121, 0.2888070046901703, 23.09076499938965, 866.337646484375, 0.28872206807136536, 0.05772889778017998, 0.10094869136810303;

  in_std_inv = in_std_inv.cwiseInverse();
  
  return 0;
}

