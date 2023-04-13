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
  

  weight0 << -1.6613e-03,  2.9875e-01,  2.5187e-01, -1.4283e-02,  2.7552e-02,
         5.6931e-02, -1.5338e-01, -9.0085e-01,  1.8734e-02,  1.1642e-01,
        -8.3405e-01,  2.7503e-02,  1.5940e-01,  3.2396e-02,  4.8040e-02,
        -4.8013e-02, -6.3321e-02, -2.8817e-02,  8.3346e-03,  3.0033e-01,
        -1.0388e+00, -1.7772e-02,  2.1279e-02,  4.9755e-02, -3.5236e-01,
         3.2577e-01,  1.2618e-02,  1.7936e-01, -3.1936e-01,  5.5088e-01,
        -6.2225e-03,  8.2509e-02,  1.4498e-01, -4.3823e-01, -3.6844e-01,
         2.6056e-02,  7.7188e-01,  2.3282e-02, -1.1100e-01,  3.9221e-02,
         2.0582e-02,  2.7035e-02, -1.3872e-01,  1.0361e-02, -3.0724e-02,
        -9.2982e-02,  2.1037e-01, -3.4862e-01,  8.0685e-03,  2.2538e-01,
         4.2486e-01, -4.5585e-02,  2.0309e-01,  7.5750e-04,  1.6994e-02,
         8.4115e-01, -2.4583e-01, -4.9474e-03, -1.4286e-02, -4.4752e-02,
         6.1155e-02, -2.7998e-01, -2.4871e-02, -2.4170e-01, -2.2719e-01,
         1.9061e-01, -5.2378e-02,  1.2238e-02,  3.8751e-02, -4.0325e-01,
        -2.0398e-01,  1.4444e-01,  2.6520e-02, -1.5750e+00, -3.2759e-01,
        -2.1180e-03,  8.6982e-03,  1.1690e-02, -1.1501e-01,  4.7229e-01,
         2.7096e-02, -3.3022e-02, -2.5503e-01,  2.8236e-01,  6.9596e-02,
        -7.7808e-02, -1.1224e-01,  4.1084e-02, -1.2363e-01, -2.8414e-02,
         1.3117e-01, -9.4351e-01, -9.0810e-02, -7.5201e-01,  1.6132e-02,
         3.3241e-02, -3.2212e-02,  1.6099e-02,  3.9548e-02, -1.3909e-02,
        -4.4695e-01,  3.4410e-01,  1.1848e-02, -2.5732e-03, -2.1053e-02,
         4.0483e-02,  1.5893e+00, -1.2145e-02,  4.6233e-02, -7.0454e-01,
         1.7692e-02,  7.2128e-01, -8.4141e-03, -3.3350e-02, -6.9630e-02,
        -8.0168e-02, -6.0510e-02, -2.9219e-01,  1.9123e-01,  3.5184e-01,
         5.5974e-02, -6.7504e-02, -1.1555e-01,  3.4392e-01,  1.7351e-01,
        -1.6681e-03,  2.3955e-02,  7.0881e-01, -2.8191e-02, -1.9427e+00,
         1.6311e-03, -2.5774e-02,  8.4298e-02,  8.6881e-02,  4.7707e-02,
        -1.1427e-01, -3.9839e-01,  5.7510e-01,  5.1219e-03,  7.7910e-02,
         1.3678e-01,  6.9259e-02, -4.2455e-01, -9.4345e-03, -1.4750e-01,
         2.5981e-01,  1.2642e+00,  8.3444e-03, -8.4655e-02, -1.5409e-01,
         4.8641e-01,  2.7156e-01, -1.1488e-02,  6.6776e-02,  8.1515e-01,
        -2.4268e-01,  2.4633e-03,  3.2144e-02,  5.3514e-02, -2.7946e-01,
         7.9550e-01, -1.0539e-02,  8.4620e-02, -1.6211e-01, -5.0591e-01,
         1.9462e-02,  9.4304e-02,  1.5287e-01, -5.0584e-01, -1.8888e-01,
         1.0185e-02,  2.3173e-01,  4.4763e-02, -6.8235e-02, -1.2244e-01,
        -1.3329e-02, -1.8674e-02,  1.9747e-01,  1.0375e-01,  2.5992e-01;
  bias0 << -1.3854, -0.5000, -1.5539, -1.0116,  1.2396, -0.4101,  1.0985, -0.5651,
        -3.4881,  1.9847, -3.0821,  3.5328, -1.0898,  0.2682, -0.7508,  0.3313,
         1.9375,  2.3199, -0.1952, -0.0676;
  weight2 <<  8.2542e-02, -5.2192e-01,  4.9023e-02,  7.0327e-02, -5.5148e-01,
        -4.7799e-02,  4.9199e-04, -5.9550e-02, -3.0897e-03,  4.6645e-01,
        -1.1572e+00,  1.0353e-02,  4.1659e-01,  4.7527e-02, -3.0998e-02,
         3.0278e-01,  4.0247e-01, -1.6378e-01,  2.5020e-01,  1.0184e-01,
         1.8612e+00,  5.9209e-01, -1.8458e+00,  1.8752e+00,  7.2376e-01,
        -1.2435e+00, -1.6683e+00,  8.0956e-01,  2.9759e+00,  3.1978e-01,
         5.2051e-01, -3.0033e+00, -2.2344e-01, -1.1647e+00, -3.5224e-03,
         9.1014e-01, -1.0217e+00,  1.9142e+00,  1.3110e+00, -1.7592e-01,
        -1.1223e+00, -1.5458e+00,  1.6082e+00, -1.2285e+00, -5.5596e-01,
         1.1468e+00,  1.0078e+00, -2.4137e+00, -2.2535e+00,  1.2840e+00,
        -3.8744e+00,  2.3919e+00,  2.6394e-01,  1.1246e+00, -3.4672e+00,
        -8.5923e-01,  7.0838e-01, -1.1379e+00, -8.7077e-01, -1.3431e+00,
        -9.7755e-02,  4.9452e-01, -5.3921e-02, -1.4845e-01, -3.8246e-01,
         1.5878e-01, -5.8056e-01, -4.0574e-02,  8.8409e-02,  8.2068e-01,
        -3.6194e-01,  2.4781e-01,  4.9120e-01,  3.0939e-02, -1.8471e-01,
         5.1513e-02, -2.4224e-01,  4.6405e-01,  2.9231e-02,  9.1535e-03,
         7.4255e-01, -5.0654e-01, -1.2962e+00, -7.1421e-01, -5.2867e-01,
         1.9279e-01, -5.7430e-02, -5.7071e-01, -7.3804e-01,  8.5981e-01,
         1.5039e-01,  5.2440e-01,  8.0245e-01,  4.4829e-01,  2.8642e-01,
         1.7096e-01,  3.8012e-01,  3.0905e+00, -7.9627e-01,  1.6277e-01,
        -2.1953e+00, -2.9480e-01,  2.3623e+00, -1.9178e+00, -6.8154e-01,
         1.3594e+00,  2.1613e+00, -3.8673e-01, -3.4415e+00, -7.5355e-01,
        -2.2882e-01,  3.6315e+00,  9.3244e-02,  7.6341e-01,  2.1930e-02,
        -1.7326e+00,  1.1943e+00, -1.9204e+00, -2.0200e+00,  6.0386e-01,
         2.7917e-03,  3.4560e-01,  2.5890e-02,  4.7976e-02,  2.0017e-01,
        -4.4892e-02, -5.7145e-02, -7.5789e-02, -1.7060e-01, -3.0560e-01,
         1.5695e+00,  6.5633e-03, -6.0461e-01, -2.2418e-01,  1.7924e-01,
         7.0019e-02,  1.0427e-01, -2.4248e-01, -1.2869e-01,  8.0848e-02,
        -7.8158e-01,  4.1602e-02,  1.2074e+00, -2.8204e-01,  8.3005e-01,
         3.5422e-01,  7.8371e-01, -9.6597e-01, -1.5980e+00,  7.6655e-01,
        -4.5281e-03,  1.5850e+00, -1.9714e-02, -7.0545e-02, -8.2469e-02,
        -1.0315e+00,  4.6763e-01, -6.8221e-01, -1.2817e+00,  7.6099e-01,
         7.5321e-01,  4.3648e-01, -1.1992e+00,  2.3848e+00,  5.5787e-01,
        -7.4132e-01, -6.4175e-01,  3.0642e-01,  2.0230e+00, -4.3156e-01,
         1.4131e-01, -1.7915e+00, -2.5563e-01, -1.2275e+00, -6.0660e-02,
        -1.7312e-01, -7.2171e-01,  9.0512e-01,  4.9061e-01,  1.1234e-01,
        -1.0855e+00, -5.0620e-01,  1.4560e+00, -7.5147e-01, -3.9181e-01,
         7.1678e-01,  1.1568e+00, -6.2839e-01, -1.5700e+00, -2.5633e-02,
        -8.4059e-01,  1.6556e+00,  7.0992e-01,  7.0605e-01, -6.3147e-02,
        -1.1277e+00,  5.0878e-01, -7.4472e-01, -8.7031e-01, -2.8658e-01,
         1.0830e+00,  3.3609e-02, -1.4878e+00,  9.9827e-01,  4.6891e-01,
        -7.4108e-01, -1.0563e+00, -4.5562e-02,  1.9463e+00, -3.6214e-01,
        -1.1847e+00, -1.8835e+00, -4.1973e-02, -1.0422e+00, -1.6908e-02,
         8.3729e-01, -5.7684e-01,  9.8068e-01,  5.2155e-01, -9.2058e-01,
         9.6958e-03,  2.2479e-02,  2.2279e-02,  1.6926e-01,  2.3577e-01,
        -3.4787e-02,  1.6403e-02, -1.5743e-01,  3.1460e-02,  8.1153e-01,
         6.2255e-01, -5.6018e-02, -1.5043e-03, -2.9769e-01, -1.0949e-02,
         8.5234e-02,  2.2795e-01,  3.2475e-02, -4.7793e-02,  6.9883e-02,
        -8.1197e-02, -1.1510e+00,  1.1234e+00,  4.1546e-01, -1.2531e+00,
         3.9513e-01,  3.1270e-01, -1.0390e+00, -1.2634e+00,  8.7896e-01,
        -4.9593e+00,  8.3340e-01,  3.7316e-01,  4.5003e-01, -2.4326e+00,
        -6.9738e-01, -1.3618e-02,  3.4523e-01, -6.5110e-02,  5.9323e-01,
        -2.1158e+00, -5.3235e-01,  1.9041e+00, -7.3573e-01, -9.8859e-02,
         1.1263e+00,  1.7947e+00, -5.9602e-01, -2.1077e+00,  2.7874e-01,
         5.4616e-01,  2.2670e+00,  6.6894e-02,  1.0094e+00,  5.2105e-02,
        -1.3516e+00,  5.7542e-01, -1.3190e+00, -1.1399e+00,  6.5272e-01,
         8.5964e-01,  1.3587e+00, -1.1445e+00,  1.1323e+00,  6.1650e-01,
        -9.6455e-01, -8.2332e-01,  1.2258e+00,  1.6605e+00, -4.5386e-01,
         8.7715e-01, -2.1928e+00, -1.0657e+00, -6.6676e-01,  7.9367e-01,
         5.3353e-01, -5.2449e-01,  8.9192e-01,  8.5593e-01,  6.5807e-01,
         1.1411e+00,  1.8730e-01, -1.6848e+00,  7.4821e-01,  4.8746e-01,
        -7.9666e-01, -1.0296e+00,  2.3194e-01,  1.6406e+00,  3.0272e-01,
         3.2551e-01, -1.6910e+00, -7.6250e-02, -1.0673e+00, -1.6130e-02,
         1.0998e+00, -2.1432e-01,  8.3728e-01,  7.1490e-01, -1.3179e-01,
        -5.5858e-01, -2.3654e-01,  1.0132e+00,  4.7500e-01,  2.4269e+00,
         3.0771e-01,  1.0091e+00, -3.0659e+00, -1.1422e+00,  1.6618e+00,
        -6.2836e-01,  1.0439e+00,  8.8057e-01,  5.0263e-01, -1.8141e-01,
        -8.1713e-01,  3.5135e-01, -3.2884e-01, -8.4614e-01,  4.7619e-02,
        -4.5875e-01, -7.9830e-01,  6.1913e-01, -1.1876e+00, -3.3047e-02,
         5.8901e-01,  7.3525e-01,  7.4416e-01, -1.1592e+00,  1.4989e+00,
         5.6060e-01,  6.0586e-01,  7.5040e-01,  7.7127e-01, -3.2857e-01,
        -3.3278e-01,  3.7264e-01, -3.8829e-01, -4.1069e-01,  2.1698e+00,
        -8.8289e-02,  7.5546e-01, -1.9661e-01,  3.1333e-02, -2.7318e+00,
         4.8655e-02, -2.8032e-04,  2.7297e-01,  1.4125e-01, -6.3169e-01,
         9.0543e-01,  2.0077e-01, -7.2539e-01,  1.2933e-01,  2.0347e-01,
        -3.6055e-01,  2.4151e-01,  5.0336e-02,  6.5332e-01, -6.0473e-01,
         8.8642e-01,  3.4600e-01, -8.4668e-01,  1.0883e+00, -3.8781e-02,
        -7.6259e-01, -9.5135e-01,  6.2657e-01,  1.6505e+00, -9.1488e-01,
        -8.5023e-01, -1.4778e+00, -2.6866e-01, -1.3013e-01, -8.6336e-02,
         7.6535e-01, -5.8445e-01,  1.0419e+00,  1.2107e+00, -6.9411e-01;
  bias2 << -0.4197,  0.1415,  1.4996,  1.6859,  0.5054, -0.5710,  0.4388,  0.2567,
        -0.6706,  0.1290, -1.1334,  0.5572,  0.9726,  0.6505, -0.8889,  0.1293,
         0.7585,  1.6321, -1.0024, -0.7873;
  weight4 << -0.8227,  0.9358, -5.1129,  2.6106,  2.8442, -3.4699,  3.1312,  1.5831,
        -0.2114, -4.0641, -3.4485,  2.0456, -5.5222,  2.8433,  3.3941,  1.8037,
         3.1578,  3.0030, -3.3548, -1.9984, -0.6021,  2.4590, -5.2310,  3.9310,
         3.1224, -2.6794,  1.9419,  1.5509,  1.1173, -2.2361, -3.2451,  1.8405,
        -4.0552,  1.2495,  5.1853,  1.4084,  2.9672,  2.7482, -2.6742, -2.1632,
         0.5686,  1.3588,  0.1642,  0.3331,  0.8678, -3.1836,  0.2257,  2.9872,
         2.3191,  0.7253,  1.6995,  0.9072, -0.0530,  2.0692, -0.2425,  1.7327,
        -0.1518, -0.1780, -0.5307, -1.8790;
  bias4 << 2.2375, 2.4605, 0.0502;
  out_std << 26.659171977493003, 39.62932133729168, 78.4321446208469;
  in_mean << 0.005049172788858414, 0.5014784932136536, 0.5003626942634583, 0.500156581401825, 60.011775970458984, 1998.331298828125, 0.7998817563056946, 0.10001819580793381, 0.34496328234672546;
  in_std_inv << 0.0028567721601575613, 0.2911893427371979, 0.2885623872280121, 0.2888070046901703, 23.09076499938965, 866.337646484375, 0.28872206807136536, 0.05772889778017998, 0.10094869136810303;
  in_std_inv = in_std_inv.cwiseInverse();
  
  return 0;
}

