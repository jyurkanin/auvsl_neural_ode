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
  Scalar slip_ratio = CppAD::abs(diff) / (CppAD::abs(tire_tangent_vel) + 1e-12);
  Scalar slip_angle = CppAD::atan(CppAD::abs(in_vec[1]) / (CppAD::abs(in_vec[0]) + 1e-12));

  // Scalar ignore1 = diff;
  // Scalar ignore2 = slip_ratio;
  // Scalar ignore3 = tire_tangent_vel;
  // std::cout << "diff " << CppAD::Value(ignore1) << "\n";
  // std::cout << "slip_ratio " << CppAD::Value(ignore2) << "\n";
  // std::cout << "tire_tangent_vel " << CppAD::Value(ignore3) << "\n";
  
  bekker_vec[0] = in_vec[3];
  bekker_vec[1] = slip_ratio;
  bekker_vec[2] = slip_angle;
  bekker_vec[3] = in_vec[4];
  bekker_vec[4] = in_vec[5];
  bekker_vec[5] = in_vec[6];
  bekker_vec[6] = in_vec[7];
  bekker_vec[7] = in_vec[8];

  // Apply scaling after calculating the bekker features from kinematics
  scaled_features = (bekker_vec - in_mean).cwiseProduct(in_std_inv);

  // Actual NN math
  layer0_out = (weight0*scaled_features) + bias0;
  layer0_out = layer0_out.unaryExpr(&tanh_scalar_wrapper);
  layer2_out = (weight2*layer0_out) + bias2;
  layer2_out = layer2_out.unaryExpr(&tanh_scalar_wrapper);
  layer4_out = (weight4*layer2_out) + bias4;        
  
  // Sign change passivity haxx
  out_vec[0] = relu_wrapper(layer4_out[0])*CppAD::tanh(100*diff);
  out_vec[1] = relu_wrapper(layer4_out[1])*CppAD::tanh(-100*in_vec[1]);
  out_vec[2] = relu_wrapper(layer4_out[2])/(1 + CppAD::exp(-100*in_vec[3]));
  
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
  
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,TireNetwork::num_in_features> temp_weight0;
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,TireNetwork::num_hidden_nodes> temp_weight2;
  Eigen::Matrix<Scalar,TireNetwork::num_out_features,TireNetwork::num_hidden_nodes> temp_weight4;
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> temp_bias0;
  Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> temp_bias2;
  Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> temp_bias4;
  Eigen::Matrix<Scalar,TireNetwork::num_out_features,1> temp_out_std;
  Eigen::Matrix<Scalar,TireNetwork::num_in_features,1> temp_in_mean;
  Eigen::Matrix<Scalar,TireNetwork::num_in_features,1> temp_in_std_inv;
  
  is_loaded = 1;
  
  weight0 <<  5.8153e-01, -8.9222e-03,  2.0581e-02, -1.6544e-03, -5.6084e-03,
        -1.6849e-02,  1.6237e-02, -1.7046e-02,  2.1246e-01,  3.2786e+00,
         1.2349e-02, -2.1270e-02, -3.0165e-02, -1.6233e-02,  4.0514e-01,
        -2.1078e-02, -1.5375e-01,  2.9609e-02,  2.9578e-01, -1.9850e-02,
        -3.9586e-02,  6.0794e-02, -1.4573e-02, -2.2998e-01, -1.1486e-01,
         4.4974e+00, -1.1447e-02, -2.0275e-02, -4.2904e-02,  1.7996e-01,
        -6.3249e-01, -1.5816e-02,  3.1694e-02,  3.3899e+00, -9.9783e-02,
        -4.2407e-02, -7.3493e-02, -2.4395e-01,  4.4048e-01, -1.6361e-02,
        -4.7303e-02,  5.0985e+00,  1.9518e-02,  2.0656e-02,  3.6091e-02,
        -1.2642e-01, -1.4892e-01, -4.8442e-03, -1.1401e-01,  2.9987e+00,
        -8.2271e-02, -5.4357e-02, -1.0268e-01,  5.2598e-01,  3.5712e-01,
        -1.0585e-02,  1.7453e-01,  8.1020e-01, -4.9401e-02,  2.0248e-02,
         3.9738e-02, -1.6364e-01,  8.1515e-03,  2.4001e-01,  2.0509e-02,
         4.0976e+00,  8.1372e-02,  4.7316e-02,  8.5339e-02, -3.1984e-01,
        -5.1236e-01, -6.6311e-04, -4.4012e-02, -1.7591e-01, -3.1427e-01,
        -8.1095e-03, -2.0869e-02,  4.0094e-01,  1.2646e-01,  5.7559e-03,
         5.6434e-02,  2.6777e+00, -4.2427e-02,  2.7560e-02,  5.6818e-02,
        -3.0182e-01, -1.6746e+00,  2.6009e-02,  6.2182e-04,  3.4826e+00,
         1.3268e-02,  8.6898e-02,  1.6343e-01, -1.7126e-01,  5.1357e-01,
        -6.4670e-03, -3.7921e-03, -4.9086e+00, -1.6125e-02,  5.3959e-03,
         1.4555e-02, -1.2508e-01,  4.7801e-01, -4.4886e-03, -8.9821e-02,
        -2.2565e-03,  3.5134e-01, -8.2811e-03, -1.4911e-02,  2.2884e-02,
         2.8325e-02, -5.5279e-03, -9.2695e-02,  7.4810e-01,  4.8960e-02,
        -1.7380e-01, -3.0286e-01,  6.3407e-01,  8.7751e-02,  4.0989e-02,
         4.5115e-02,  2.7729e+00, -2.2732e-02, -3.4031e-02, -5.5543e-02,
         1.2739e-01,  7.1903e-01,  4.2014e-03;
  bias0 <<  1.1670, -0.0761,  1.1250,  1.6990, -0.1322,  1.0497,  1.1688, -0.8269,
         0.8140, -0.4409, -3.3976, -0.1474, -1.3584,  0.4444, -0.2566,  0.2182;
  weight2 <<  3.8107e-01,  4.4545e-02, -2.0297e-02, -1.3798e-01, -2.3449e-02,
         1.0743e-01, -1.4515e-01,  1.0994e-02, -1.1708e-01, -6.3826e-03,
         8.6098e-02, -8.3946e-02, -2.0970e-01,  9.0927e-01, -3.1496e-02,
        -1.9650e-05, -9.3516e-01,  5.5659e-01,  8.5463e-02,  2.1083e+00,
         1.4808e+00, -6.7379e-02,  1.2902e+00,  7.4817e-02,  1.1551e+00,
         7.3869e-01, -2.8347e+00,  1.3986e+00, -1.1654e+00,  5.3894e-01,
         5.8826e-01,  1.2857e+00, -7.4204e-01, -1.6644e-01,  1.2356e-03,
         1.5962e-01,  1.5751e-01, -2.8150e-02,  1.6516e-01, -4.7960e-02,
         1.6679e-02, -4.0455e-02, -9.5855e-03,  2.4963e-02,  2.2343e-01,
        -8.1463e-01,  5.1334e-02, -4.7837e-02,  7.0561e-01, -1.4464e+00,
        -3.0266e-01, -2.7072e+00, -1.8023e+00, -7.2390e-01, -2.8821e+00,
        -4.7252e-01, -2.8113e+00, -9.1398e-01,  1.2706e+00, -2.0837e+00,
         2.8604e+00, -7.0827e-01, -1.0194e+00, -2.0342e+00, -1.9396e-01,
        -6.3652e-01, -4.0181e-01, -1.0885e+00, -7.2876e-01, -3.2776e+00,
        -1.4009e+00,  4.0345e-01,  5.0084e-01, -5.9017e-01, -4.6386e-01,
        -9.1019e-01,  1.8788e+00, -1.4825e-02, -7.2599e-01,  6.8290e-01,
         1.0257e+00, -1.0509e+00, -1.3913e+00, -4.8868e-01, -6.5987e-01,
         5.0522e-03,  3.8596e-01, -3.0430e-01,  8.8469e-02, -8.8946e-01,
         1.9453e+00, -7.0415e-01,  6.6253e-01, -1.2970e+00, -3.0547e-01,
         3.3312e-02,  4.9198e-01, -1.8475e+00,  5.7890e-01, -2.7368e+00,
        -1.4790e+00, -3.1429e+00, -2.0503e+00,  1.2150e+00, -2.5373e+00,
         1.4388e-01,  1.3151e-02, -2.1258e+00,  2.8056e+00,  1.8721e-01,
        -8.1925e-01, -2.7180e+00, -7.9073e-01,  2.4863e+00,  2.1101e-01,
         1.2738e+00,  9.7938e-01,  3.0375e+00,  2.5478e+00,  9.4808e-01,
         2.4347e-01,  7.1074e-02, -3.2585e+00,  1.3124e+00, -2.7233e+00,
         3.3762e-01,  4.7082e-01,  8.5941e-01,  8.1616e-01, -4.8140e-01,
        -4.1712e-01, -1.5629e+00, -6.7441e-01, -1.7458e+00, -2.3001e+00,
        -2.8347e+00, -6.5769e-01, -4.3525e-02,  2.1601e+00, -9.0296e-01,
         7.0247e-01, -5.9529e-01, -4.9705e-01, -7.3089e-01,  7.6928e-01,
        -1.1362e+00,  1.6097e+00, -1.5446e+00, -1.1055e+00, -1.6163e+00,
        -2.6772e+00,  2.6659e-01, -2.2531e+00,  8.1054e-03,  2.3770e+00,
        -1.3476e+00,  2.1506e+00, -4.7804e-02, -9.9385e-01, -1.7035e+00,
         1.6843e+00,  1.8250e+00, -2.8893e-01,  6.3947e-01,  4.5359e-01,
         6.2679e-01,  1.4165e+00,  1.3074e-01,  5.9702e-01, -1.1091e-01,
        -1.8085e+00,  1.4353e+00, -2.7031e+00,  2.5036e-02,  8.4456e-01,
         7.7845e-01,  8.2232e-02,  9.5813e-01,  4.3923e-01,  1.8446e+00,
         9.5151e-01,  3.2380e+00,  1.8963e+00,  2.3232e-01,  1.3325e+00,
         2.7524e-01,  5.5627e-01,  1.1123e+00, -1.7701e+00, -1.2913e-01,
         6.6781e-01,  1.5515e+00,  6.6195e-01,  3.2443e-01,  1.0003e-01,
         5.9335e-01,  4.9773e-01,  7.2456e-01, -3.4601e-01, -4.2405e-02,
        -6.2304e-01, -1.2841e-01,  5.3056e-01, -1.0611e+00,  1.1353e+00,
        -6.5637e-01,  7.1132e-02, -6.1781e-01,  6.5621e-01,  6.0866e-01,
        -1.1566e-01, -1.7557e+00, -1.2282e+00, -1.1552e+00,  7.8382e-01,
         4.7037e-01,  1.3720e-01, -5.5226e-01,  2.8567e+00, -1.4320e+00,
         5.7869e-01, -1.1083e-01, -6.6510e-01, -1.0694e+00, -1.2913e+00,
        -4.1429e-01,  3.3828e-02,  2.6860e-01,  3.0979e-01, -1.2043e-01,
         2.4772e-01, -6.6647e-02,  1.3447e-01, -6.0187e-02, -7.5953e-03,
         4.4295e-02,  4.6131e-01, -1.0044e+00,  1.0367e-01, -3.9833e-02,
         9.1324e-01, -1.9112e+00, -4.4108e-01, -7.0044e-01, -8.8252e-01,
        -2.6913e+00, -2.8021e+00, -4.3520e-01, -3.8757e-01, -1.1991e-01,
         1.2647e+00, -1.4413e+00,  1.6376e+00,  1.9971e-01, -1.2909e+00,
         2.5825e-01;
  bias2 << -1.0782, -0.1844,  0.6866, -0.3522, -0.8495, -1.4437, -1.0693, -0.4948,
        -0.6029, -0.1111,  0.0269,  0.7950,  1.7196,  0.1638,  0.5661, -1.1151;
  weight4 << -2.8187, -2.0256, -0.5025,  2.8698, -0.2295,  1.4845, -1.0648,  0.6501,
        -0.9572, -3.4006,  0.6811, -0.3954,  0.2824,  1.0820, -0.8063,  0.7430,
        -1.1269, -1.3702, -2.4389,  2.0946, -0.2859, -1.7817,  1.8521,  1.6340,
        -1.8248,  1.9309,  0.9442,  1.6427,  0.0747,  2.0572,  0.8624, -0.9951,
         0.2299, -2.2725,  0.1701,  4.2036, -1.7120, -0.3131, -0.0143,  3.5022,
         0.8210,  0.3116,  1.8269, -1.5362,  1.5191,  1.9782,  0.0173, -1.8169;
  bias4 << 0.1386, 0.2830, 0.5928;
  out_std << 26.659172778230566, 39.62932253025853, 78.43214649313944;
  in_mean << 0.005049172323197126, 35.31148910522461, 0.784349262714386, 60.011775970458984, 1998.331298828125, 0.7998817563056946, 0.10001819580793381, 0.34496328234672546;
  in_std_inv << 0.0028567721601575613, 183.33016967773438, 0.41536253690719604, 23.09076499938965, 866.337646484375, 0.28872206807136536, 0.05772889778017998, 0.10094869136810303;
  in_std_inv = in_std_inv.cwiseInverse();
  
  return 0;
}

