#include "TireNetwork.h"
#include "generated/model_constants.h"
#include <cmath>
#include <cppad/cppad.hpp>
#include <iostream>



TireNetwork::TireNetwork()
{
  out_std << 44.45929437637364, 44.319044796830426, 55.11481922955709;
  in_mean << 0.005, 0.0, 0.0, 0.0;
  in_std_inv << 0.0028585679829120636, 0.5799984931945801, 0.576934278011322, 0.5774632096290588;
  in_std_inv = in_std_inv.cwiseInverse();  
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
void TireNetwork::forward(const Eigen::Matrix<Scalar,8,1> &in_vec, Eigen::Matrix<Scalar,num_out_features,1> &out_vec, int ii){
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
  
  // Apply scaling after calculating the bekker features from kinematics
  scaled_features = (bekker_vec - in_mean).cwiseProduct(in_std_inv);
  
  // Actual NN math
  layer0_out = (m_params[ii].weight0*scaled_features) + m_params[ii].bias0;
  layer0_out = layer0_out.unaryExpr(&tanh_scalar_wrapper);
  layer2_out = (m_params[ii].weight2*layer0_out) + m_params[ii].bias2;
  layer2_out = layer2_out.unaryExpr(&tanh_scalar_wrapper);
  layer4_out = (m_params[ii].weight4*layer2_out) + m_params[ii].bias4;
  
  // Sign change passivity haxx
  out_vec[0] = relu_wrapper(layer4_out[0])*(1*diff);
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
	    m_params[0].bias4.size());
}

void TireNetwork::setParams(const VectorS &params, int idx)
{
  assert(params.size() == getNumParams());
  
  for(int kk = 0; kk < num_networks; kk++)
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
    
  for(int kk = 0; kk < num_networks; kk++)
  {
    m_params[kk].weight0 <<
      1.3141e-01, -1.2987e+01,  1.5245e+00, -4.6593e-01,  3.0292e-01,
      -2.2036e+00, -3.2484e-01, -2.9473e-01, -1.5334e-01,  1.6665e+00,
      3.1269e-01, -2.3139e-01, -4.9404e-01,  7.9944e-02, -1.8339e-02,
      3.5794e-01, -9.9548e-02,  1.7324e+00,  2.3429e-01,  1.9648e+00,
      9.4129e-01,  1.1309e-01, -3.8214e-02,  1.1511e-01, -6.0517e-02,
      1.3662e+00,  1.9678e-01, -1.7009e+00,  1.4794e-01, -1.7548e+00,
      3.6236e-01, -5.8246e-01,  4.2608e-01,  5.7721e-02, -2.0646e-02,
      1.0250e-01, -2.9338e-01,  9.3208e+00, -2.0966e-01,  3.5583e+00,
      -1.7692e-01,  1.6318e+00,  5.7103e-01, -6.4896e-01,  5.5845e-02,
      -4.3356e-01, -9.4772e-01, -7.6354e-01,  5.0339e-02, -9.5673e-01,
      -1.3465e-01, -1.6171e-01,  1.1141e-02,  6.1717e-01,  1.3385e-01,
      -1.5825e+00,  3.0578e-01, -7.1654e-02,  2.9459e-02, -1.3127e-01,
      -1.9765e-01,  8.0084e-01,  5.3346e-02,  8.9352e-01;
    m_params[kk].bias0 <<
      0.0019, -0.1881,  0.4557,  1.0243,  0.1117,  1.1642, -0.6209,  0.0561,
      -0.5864, -0.1359, -0.7462, -1.2327,  0.6848,  0.4452,  0.5220,  0.0245;
    
    m_params[kk].weight2 <<
      -2.5459e-01,  4.0520e-01,  7.7315e-01, -1.7542e-01,  7.1213e-01,
      -1.4372e-01, -8.3882e-01,  4.4784e-03,  1.6307e-01,  3.6531e-01,
      -5.0149e-01, -5.9684e-01,  6.1345e-01,  2.5684e-01,  1.9869e+00,
      7.2812e-01, -2.4815e-01,  1.7479e-01,  5.4650e-01,  2.6766e-01,
      9.1368e-01, -8.0482e-01, -1.8846e+00, -5.4987e-01,  1.9832e-01,
      -1.0177e-01, -4.1753e-01, -1.0760e+00, -1.8043e-03, -9.3674e-01,
      -1.2439e-01,  1.5491e+00,  1.7037e-01, -3.6966e-01,  2.5115e-02,
      5.2826e-01, -7.8785e-02, -4.3776e-01,  7.7420e-02, -3.0921e-01,
      -9.6615e-02,  2.8540e-01,  5.0285e-02,  3.3042e-01, -4.0301e-01,
      -2.5102e-02, -3.9386e-01, -6.0380e-02, -1.6743e-01, -3.3347e-01,
      -2.3546e-01, -4.7363e-01,  1.1922e-01,  3.1930e-01, -4.2890e-02,
      9.9367e-02,  4.8951e-01,  9.1904e-01, -4.0632e-02,  5.3695e-01,
      -3.5236e-02, -3.9346e-02,  6.2710e-01,  3.3033e-01,  1.2698e-01,
      1.6337e-01,  1.4175e-01, -3.6644e-01, -1.2805e-01,  2.4674e-01,
      3.5079e-02, -9.7356e-02,  8.3592e-01, -1.7763e-01,  2.3685e-02,
      -2.4510e-01, -1.4501e-02,  6.4514e-03,  4.2411e-01, -1.2114e-01,
      -1.0760e+00,  4.1695e-01,  1.3472e+00, -1.8257e-01, -1.3808e+00,
      4.7379e-01, -3.2319e-01, -2.7367e-01,  9.5786e-02, -8.9432e-01,
      -6.9171e-02,  1.7032e-01,  4.0632e-01,  2.5330e-01,  3.1886e-01,
      -8.1938e-01, -1.8433e-02,  2.1581e-01,  1.7356e-01,  5.0356e-01,
      2.7432e-01, -3.0171e-01, -4.5281e-02, -2.0625e-02, -1.5726e+00,
      7.1427e-01, -1.9395e-02, -5.1715e-02,  9.1096e-02,  7.9244e-02,
      -7.5144e-01, -3.0202e-02,  2.2048e-01,  1.4435e-01,  1.3120e-01,
      8.8758e-02,  2.5754e-01,  6.7903e-01,  8.6883e-02, -3.1943e-02,
      -1.9447e-01,  1.7662e-02,  1.3285e-01, -8.1337e-01,  1.2595e-01,
      -1.6612e-01,  1.7491e+00,  8.5637e-02,  5.8468e-01,  3.7359e-01,
      -4.7993e-01, -2.3405e-02, -5.3780e-01,  1.0369e-01, -4.2102e-01,
      9.8577e-01, -8.4415e-02, -1.2483e+00, -4.2293e-01, -6.7628e-02,
      1.0071e+00,  5.3444e-01,  3.3336e-01,  1.6895e-01, -2.2518e+00,
      1.6451e+00, -7.4779e-01,  8.9917e-02, -8.6514e-02,  3.9921e-01,
      -2.4047e-02, -6.0281e-01,  3.7166e-01,  7.5023e-01, -1.7971e-01,
      8.1824e-01,  3.1470e-01, -1.9320e-01,  5.9800e-03, -6.4031e-01,
      1.3017e-01,  1.7203e-01, -4.9412e-02, -2.3939e-01,  7.1815e-01,
      1.8773e+00,  3.2394e-01, -5.3516e-01,  4.4051e-01,  4.3200e-01,
      -9.4709e-02, -5.7888e-01, -1.6987e-01, -3.3855e-01,  4.0085e-01,
      -9.0468e-02, -1.4098e-01,  1.1041e-01,  5.3749e-01,  5.7625e-01,
      2.1771e-01, -6.7319e-01,  6.9078e-02, -4.6333e-01,  3.9574e-02,
      4.1659e-01,  6.9370e-01,  2.8043e-01, -1.0134e+00, -3.1241e-01,
      -3.1169e-01,  4.3205e-02,  1.6046e+00, -1.4959e-01, -1.1170e+00,
      2.3965e-01, -6.9752e-01, -1.5868e+00,  5.8696e-01,  6.3058e-03,
      -2.0457e-02, -5.2757e-02, -3.5212e-01,  1.1890e+00,  2.7462e-01,
      -1.0484e+00, -1.4235e+00, -8.9033e-01, -2.2869e+00, -2.7393e-01,
      2.4619e-01, -3.3719e-01, -7.2917e-01,  7.8680e-02, -2.6065e+00,
      3.7160e-01, -4.0177e-01, -4.3013e+00, -1.0544e+00,  6.5849e-01,
      4.9187e-01,  3.4801e+00,  1.6394e-01, -4.9914e-01, -9.9521e-01,
      6.2420e-01,  1.0884e+00, -1.1382e+00,  2.1031e+00, -7.0902e-01,
      -2.5625e-01, -6.8042e-01,  1.6902e+00,  3.9522e-01,  2.2401e-01,
      -8.9760e-01,  1.8714e-01, -1.0242e-01,  1.4447e+00,  3.0575e-02,
      -9.5088e-02,  5.8086e-02, -8.9465e-01, -6.1254e-01, -2.7702e-01,
      4.1242e-01, -5.7342e-01,  4.2988e-01, -3.0814e-01, -1.3136e+00,
      -9.5937e-01, -4.5984e-01,  1.5678e-01,  1.3767e+00,  3.2374e-01,
      3.2528e-02;
    m_params[kk].bias2 <<
      0.8748,  0.8798, -0.2461, -0.8584,  0.5045,  0.1448,  0.2913,  0.7450,
      0.0497, -0.9702,  0.6085, -0.0963, -1.0628, -0.7158,  1.0474, -0.0952;
    
    m_params[kk].weight4 <<
      1.5042,  2.0114, -2.5168, -1.2710,  1.4424,  0.9733,  0.2066,  1.1662,
      1.4453, -3.3685,  1.7582, -2.1479, -1.5519,  3.2961,  1.7787,  3.6431,
      1.4211,  1.6075, -0.6801, -1.2871,  1.4821,  2.4899, -0.0834,  1.4713,
      1.7626,  0.1480,  1.2422, -0.4754, -1.6544,  2.9762,  1.6294,  1.4808,
      0.8362,  0.6830, -0.2246,  1.4721,  2.3837,  0.0209, -1.2657,  0.8697,
      0.1354,  0.0180,  0.0466,  0.0211, -0.8208, -0.0434,  0.8866,  0.0908;
    m_params[kk].bias4 << 0.8708, 1.3100, 0.3692;
  }
    
  return 0;
}

//
/*
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
