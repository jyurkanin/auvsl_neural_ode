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
  

  weight0 << -7.2295e-02,  2.0413e-01,  9.2996e-01, -1.6524e-03, -5.9510e-02,
        -1.2024e-01,  2.5315e-01,  3.2856e-01, -4.0900e-02, -6.9923e-02,
        -8.0489e-01, -5.8645e-03,  8.3474e-01, -1.0013e-02, -2.9453e-02,
         4.1280e-02, -4.1969e-03, -3.1739e-02,  7.2168e-02,  1.0643e+00,
         3.4611e-02, -2.6807e+00,  2.6378e-02,  4.2920e-02, -5.3957e-02,
         5.0689e-02,  9.2788e-02,  9.5488e-02,  3.4717e-01, -7.7399e-01,
        -1.1751e-02,  1.7776e-02,  4.5202e-02, -4.2381e-01,  2.9376e-01,
        -1.7887e-02, -2.0145e-02, -3.5379e-01,  9.8345e-01,  4.0744e-03,
         2.8657e-02,  5.8817e-02, -9.2267e-02, -3.3196e-01, -2.7973e-02,
        -1.1954e-01,  3.1078e-01, -6.4315e-01, -7.4066e-03, -8.0245e-02,
        -1.5009e-01,  4.4062e-01,  3.9638e-01, -4.1130e-02, -3.5207e-02,
        -4.1646e-01,  1.9176e-01,  1.6096e-03, -1.7403e-02, -4.1634e-02,
         1.7854e-01,  1.4014e+00,  6.0043e-03,  8.7112e-02, -3.8723e-01,
        -3.4034e-01,  1.6443e-02,  1.0777e-01,  1.9968e-01, -4.1792e-01,
        -4.5761e-01,  4.1732e-02,  2.3970e-01, -3.2097e-02,  6.5869e-02,
         1.5396e-02,  3.4030e-02,  6.2667e-02, -2.6337e-01, -1.1585e-01,
         2.5526e-01,  2.6255e-02, -1.3152e+00, -9.7142e-02, -1.3033e-02,
         7.6058e-03,  1.7386e-02, -1.2918e-01,  3.5361e-01,  1.0134e-04,
        -2.4437e-02,  3.0339e-01, -1.7322e-01,  1.0446e-02, -5.7635e-02,
        -1.1562e-01,  5.2395e-01,  3.4611e-01,  1.3067e-01,  9.9626e-02,
        -8.3537e-02,  1.7110e-01,  1.0468e-02, -2.8393e-01, -5.0844e-01,
        -6.1878e-02, -1.3439e-01,  1.7600e-02, -4.9729e-02,  1.1354e+00,
        -1.3978e-01, -8.1681e-02, -1.0738e-02, -1.0229e-02,  4.2876e-02,
         2.3042e-01,  1.2179e-03,  2.6492e-01,  3.9918e-02, -6.2159e-02,
        -4.0602e-02,  3.5808e-02,  6.7240e-02,  2.6226e-02, -9.0133e-03,
         6.8360e-03,  1.0521e+00,  1.5655e-02, -6.6635e-02,  1.1534e-02,
        -1.7626e-02, -2.4233e-02,  2.8456e-02,  2.4017e-02, -4.7638e-03,
        -6.7091e-02, -6.9209e-01,  2.2785e-01,  6.5823e-03, -7.5058e-02,
        -1.3286e-01,  3.1430e-01, -3.7410e-01, -2.4149e-02;
  bias0 <<  1.2538, -0.1099, -3.3775, -1.3423,  0.8451,  1.1252,  3.1027, -0.0460,
        -1.1740, -2.7602,  1.0756,  0.4050,  1.3439, -0.2563,  2.0109, -1.5763;
  weight2 <<  9.4821e-01, -3.0691e-01,  3.3218e-01,  2.0447e+00, -6.5291e-01,
         1.9196e+00,  3.6676e+00, -1.6778e+00,  1.5031e-01, -4.1227e+00,
         9.7280e-01, -7.6848e-01,  4.8243e-01,  1.8028e-01, -8.8313e-01,
         2.1245e+00,  8.3812e-02, -4.2998e-01,  9.6509e-02, -1.3192e-02,
        -3.8266e-02, -1.4075e-01, -1.9984e-03,  4.9802e-02, -5.8371e-01,
         2.7311e-01, -2.1308e-01,  6.9377e-02, -1.5164e-01,  5.5327e-01,
         6.7980e-01,  1.0889e-01,  1.0448e+00,  2.3072e-02,  3.5722e-02,
         2.3304e+00, -5.3987e-01,  2.0232e+00,  3.7824e+00, -1.7803e+00,
        -2.2849e-01, -4.2885e+00,  1.0739e+00, -7.0485e-01,  2.8572e-01,
         3.3285e-01, -1.1366e+00,  2.0969e+00,  3.8131e-01, -4.0935e-02,
        -5.8861e-02,  9.3262e-01, -5.5205e-01,  7.7622e-01,  1.9956e+00,
        -5.6640e-01,  1.3065e+00, -1.9541e+00,  9.8471e-01, -4.3255e-01,
         1.8626e-01, -4.6690e-01, -4.1794e-01,  9.7573e-01, -1.0047e-01,
        -1.3810e-01,  2.2090e-01,  6.4218e-02,  1.3057e-01,  1.1414e-01,
         6.3689e-01,  9.4215e-02,  8.0634e-01, -3.2069e-01,  1.0543e+00,
        -7.2218e-02, -2.2171e-01, -1.0204e+00, -6.9369e-01,  6.7118e-02,
        -6.7118e-01, -8.2256e-01, -1.8727e-02, -9.7216e-01,  7.3629e-01,
        -6.5742e-01, -1.1830e+00,  8.0354e-01,  1.0486e+00,  8.3625e-01,
         5.0169e-01,  2.4729e-01, -6.5600e-01, -2.3478e-01,  1.0453e+00,
        -6.2787e-01,  8.0506e-01,  1.8176e-02, -9.7887e-03,  1.6719e+00,
        -9.3638e-02,  1.5245e+00,  2.5907e+00, -1.2257e+00, -1.5500e-01,
        -3.0716e+00,  8.3102e-02, -6.1403e-01,  1.2527e-01, -2.8551e-01,
        -1.2429e+00,  1.0399e+00, -6.0073e-01, -1.4268e-01, -1.1774e+00,
        -9.1719e-01,  4.4823e-01, -5.8561e-01, -1.2685e+00,  7.3924e-01,
         1.2057e+00,  1.6536e+00,  2.3806e-01,  2.6273e-01, -7.0356e-01,
        -1.7517e-01,  1.0263e+00, -5.7334e-01,  7.9866e-02, -2.8532e-02,
        -3.4732e-02, -1.3797e+00,  2.6251e+00, -6.3829e-01, -2.3075e+00,
         1.7248e+00, -8.4186e-02,  2.4037e+00,  1.5271e+00,  3.7927e-01,
         6.1574e-01, -1.4438e-01,  6.5559e-01, -1.2220e-01, -5.9587e-01,
        -6.1284e-01,  8.4869e-02, -1.0962e+00,  7.6535e-01, -2.6557e-01,
        -5.0880e-01,  2.1962e-01,  6.3107e-01,  3.2762e+00,  1.4226e-01,
        -2.4672e-02, -2.0377e+00, -1.7999e-01, -1.0675e-02,  9.4996e-01,
        -8.5487e-01, -1.1608e-02,  3.3284e-02, -1.2867e+00,  1.4449e+00,
        -1.4779e+00, -2.7732e+00,  1.2589e+00, -2.8119e-01,  3.0190e+00,
        -1.5087e-01,  5.7759e-01, -2.0506e-01, -1.9951e-01, -2.3914e-01,
        -1.1702e+00,  6.4524e-01,  1.5506e-02,  1.7887e-02,  1.4800e+00,
        -1.6372e+00,  1.0846e+00,  2.6628e+00, -9.5516e-01,  2.3497e-01,
        -2.5205e+00,  3.3090e-01, -5.8837e-01,  3.1398e-01, -4.6640e-01,
        -7.4029e-01,  1.0276e+00,  7.1243e-02,  4.0531e-01, -1.2063e+00,
         4.3595e-02,  1.9907e-03, -8.5003e-02,  4.0456e-01,  2.0082e-02,
        -2.5657e-02,  3.8056e-01,  1.0823e-01,  4.1796e-02, -5.6470e-01,
         7.0088e-01,  9.7637e-01, -1.7451e-01, -9.3985e-02, -5.9622e-02,
        -3.8921e-02, -1.3499e+00,  2.8196e+00, -1.5018e+00, -1.8468e+00,
         1.8105e+00, -9.0959e-01,  2.4737e+00,  1.9940e+00,  4.5792e-01,
         8.2726e-01, -1.5230e-01,  7.6979e-01, -2.8889e-01, -3.0591e-02,
         1.1678e-01, -6.0563e-02, -9.6961e-02, -1.3353e-01,  6.5196e-02,
        -3.3678e-01, -9.0133e-02, -3.5714e-01,  4.6946e-02, -5.8472e-01,
        -4.8771e-02,  1.4897e-01, -5.6912e-01, -5.8072e-01,  2.4094e-02,
         3.3900e-01, -7.2020e-01, -1.4605e+00,  3.7940e-01, -8.3578e-03,
         1.5156e-01,  5.1548e-01, -1.8634e-01, -3.4501e-01, -1.3298e+00,
         5.6680e-01, -1.8633e-01,  9.2927e-01, -5.2720e-02, -8.6703e-01,
         6.5414e-01;
  bias2 <<  0.0882, -1.2068, -0.5941,  0.7057, -0.3128, -2.3057,  0.5912, -2.1949,
         1.4820, -0.0063,  0.0596, -0.2393, -2.6946,  0.4796,  0.7947,  0.2061;
  weight4 <<  1.6942,  0.1635, -2.1829,  1.2480, -0.3334,  2.4560, -1.7216, -2.5644,
        -0.3572, -1.4615, -0.6418, -0.6613, -0.7385,  0.3409, -0.7048, -0.2518,
        -1.0701, -0.6206, -0.7882,  1.7276, -0.2919, -0.9954, -1.8233,  1.9748,
        -0.2489, -0.2050, -1.4207, -0.7201,  0.3900,  0.2517, -0.3131,  1.1122,
         0.0604,  0.2286, -2.9384, -0.6669, -0.0079, -0.1554, -2.6559, -0.1300,
         1.0508, -0.3698, -3.5874, -1.2449,  0.1253,  1.8612,  0.0434, -0.1647;
  bias4 << -0.0835,  1.2038,  0.7808;
  out_std << 26.659172778230566, 39.62932253025853, 78.43214649313944;
  in_mean << 0.005049172323197126, 0.5014784932136536, 0.5003626942634583, 0.500156581401825, 60.011775970458984, 1998.331298828125, 0.7998817563056946, 0.10001819580793381, 0.34496328234672546;
  in_std_inv << 0.0028567721601575613, 0.2911893427371979, 0.2885623872280121, 0.2888070046901703, 23.09076499938965, 866.337646484375, 0.28872206807136536, 0.05772889778017998, 0.10094869136810303;
  in_std_inv = in_std_inv.cwiseInverse();
  
  return 0;
}

