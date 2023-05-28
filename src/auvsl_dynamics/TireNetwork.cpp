#include "TireNetwork.h"
#include "generated/model_constants.h"
#include <cmath>
#include <cppad/cppad.hpp>
#include <iostream>



TireNetwork::TireNetwork()
{
  out_std << 44.45929437637364, 44.319044796830426, 55.11481922955709;
  in_mean << 0.005049827974289656, 0.5013627409934998, 0.49966344237327576, 0.5000784397125244, 0.05001185089349747;
  in_std_inv << 0.0028585679829120636, 0.2916049659252167, 0.2884257733821869, 0.288765013217926, 0.02888154424726963;
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
    m_params[kk].weight0 <<  2.4305e-01, -6.5279e-01, -2.6631e-01,  5.6838e-01,  6.4865e-03,
      5.6002e-01,  1.9517e-01, -5.8955e-02,  3.4654e-02, -2.6065e-03,
      4.2070e-02, -1.0348e+00,  1.1308e-01,  1.4839e-01,  5.8081e-03,
      -4.2602e-01, -1.5916e+00,  4.6987e-01, -4.1107e-02,  4.2229e-04,
      -1.5963e-01, -9.7215e-01,  2.3955e-01,  4.9586e-01, -9.6850e-03,
      5.7797e-01, -1.8876e-01, -3.2707e-02,  6.2549e-02,  2.2754e-03,
      -3.7747e-01,  1.0500e-01, -1.4010e-01,  8.2574e-02,  3.7913e-03,
      -5.6882e-01, -1.6784e-01,  5.6750e-02, -7.1950e-03, -1.0245e-03,
      4.0670e-02,  1.9877e+00, -2.2311e-01, -4.7446e-01,  9.8732e-03,
      3.4549e-01, -5.8986e-01, -1.8618e-01, -1.5619e-01,  1.3400e-02,
      -1.7528e-02,  1.1645e+00, -6.3842e-02, -4.4935e-03, -1.2199e-02,
      7.1094e-02, -3.8515e-01, -3.5735e-01, -5.3999e-01, -2.1520e-02,
      -2.0327e-02, -5.6958e-01, -1.7799e-01,  1.5241e+00, -1.5386e-02,
      3.2847e-01, -8.2157e-02,  5.3974e-02, -7.6216e-02, -1.4492e-03,
      -5.9242e-02,  9.7544e-02, -2.6086e-02,  5.3523e-01,  8.3575e-03,
      -3.1658e-01, -1.0477e+00,  4.8211e-01,  1.0604e+00, -4.3186e-03;
    m_params[kk].bias0 <<  0.4882,  0.3566,  0.0754, -1.4490, -0.7566,  0.4812, -0.8026,  2.1776,
      1.0252, -1.3531,  2.1105, -1.7060,  0.8149, -0.6266, -0.3988,  0.5371;
    m_params[kk].weight2 << -3.6005e-02, -1.5481e+00, -2.0176e-02, -2.5593e-02,  1.3693e-01,
      -2.6523e-02,  3.0979e-01,  1.4263e+00,  3.2212e-02, -1.1098e-01,
      -4.9407e-02, -5.4127e-02,  2.9991e-06, -6.5414e-01, -2.1755e-01,
      6.6591e-03,  1.1186e-01, -1.9374e+00,  3.6011e-01, -1.2762e-01,
      -5.3129e-01, -1.5196e+00,  4.6011e-01, -1.1939e+00, -2.8345e-01,
      -1.0930e-01, -2.4557e-01,  1.2665e-01,  7.8994e-02,  2.7854e-01,
      -1.3285e-01,  2.7107e-02,  8.8792e-02,  5.8999e-01,  2.4734e-01,
      -4.0677e-02, -2.6884e-01,  4.3369e-01, -4.6599e-01, -6.9922e-01,
      -5.9342e-02,  1.7159e-01, -5.3708e-01,  1.8820e-01, -1.5925e-02,
      5.2119e-01,  1.4059e-01, -9.9793e-02,  8.2269e-02,  1.3609e+00,
      -2.7038e-01, -7.7366e-02, -1.1165e-01, -3.3064e-01, -2.4945e-01,
      8.0737e-01,  1.8041e-01,  7.3204e-01,  7.7010e-01, -5.4078e-01,
      2.4694e-01, -1.1246e-01,  1.8154e-01,  1.9532e-01,  8.9473e-03,
      -4.4650e-01,  3.0636e-02,  4.9575e-02,  7.4774e-02, -3.3977e-01,
      3.4447e-01, -1.8217e-02, -9.1664e-03,  4.6439e-02,  6.7699e-02,
      5.6419e-02, -1.8116e-02, -8.1198e-01, -1.2211e-01,  4.4576e-02,
      1.1441e-01, -9.0042e-02, -2.9240e-01,  5.6023e-01, -4.1384e-01,
      -3.7413e-02,  1.1783e+00, -6.3682e-01, -2.0166e-01, -1.7631e+00,
      -1.5043e+00,  7.1525e-01, -3.2252e-01,  7.8610e-01, -1.5837e+00,
      4.2533e-01,  3.4825e-01, -3.0026e-01, -5.4263e-01,  4.9606e-01,
      -2.6963e-01, -1.1907e+00, -1.1872e-01, -8.2264e-01,  4.0442e-01,
      -4.8325e-02, -7.0588e-01, -2.4276e-01, -1.9966e-01,  8.4508e-01,
      7.1635e-01, -1.8273e-01, -1.4250e+00,  1.0728e+00,  3.0940e-01,
      4.6107e-01, -1.5268e+00, -6.9342e-01, -7.0194e-01,  5.2929e-01,
      2.0473e+00, -1.5163e+00,  5.8002e+00, -1.3313e+00,  3.4652e+00,
      -3.8755e-01,  2.2788e-01,  1.7124e+00, -7.8261e-01,  6.0802e-01,
      -2.9406e-01, -1.5635e-01,  2.2940e-02, -3.5653e-01,  2.2296e-01,
      5.1376e-01,  4.9349e-01, -6.7182e-01,  2.4958e+00,  2.2753e-01,
      6.5020e-01,  1.0228e-01,  9.0626e-02,  1.3265e+00, -6.8689e-02,
      -3.6230e-01, -2.1710e-01, -3.9548e-03,  4.8998e-02, -2.9938e-01,
      4.6242e-02,  5.7440e-02,  3.0120e-02, -1.9324e-01, -2.2916e-02,
      -2.2982e-01,  5.4600e-02, -3.4137e-01,  1.6876e-01, -3.5744e-02,
      -1.0461e+00, -4.1216e-01, -5.3790e-01,  2.2156e-01,  5.5630e-01,
      -1.8689e+00,  4.7841e-01, -1.4093e+00,  5.5126e-02,  9.2949e-02,
      -1.1628e+00, -7.6431e-02,  4.7370e-01,  4.3941e-01,  3.9121e-01,
      -4.2412e-01,  1.3033e+00, -1.5520e-01,  3.1000e-01, -1.3109e-01,
      -5.1482e-01,  4.6996e-01, -1.8349e-02,  8.1600e-01,  5.0452e-01,
      8.5894e-01,  1.0079e+00,  1.7705e-01,  1.6690e-01, -1.0785e+00,
      -4.1266e-01,  1.8761e-01, -1.1231e+00, -1.3007e-01,  6.3835e-01,
      -2.7960e+00, -8.5070e-01, -2.6771e-01,  3.1933e-01, -5.6114e-01,
      -8.6924e-01, -1.9109e+00,  2.0838e+00, -4.7409e-01,  5.1545e-01,
      -1.2229e+00, -2.0014e-03, -1.9924e-01, -9.4903e-02,  5.2118e-01,
      -2.6871e-01,  6.7408e-02,  3.7712e-01,  3.6479e-01,  9.4960e-02,
      -8.2475e-01,  3.2676e-02, -3.5391e-01,  8.3136e-02, -4.1774e-01,
      1.5405e-02,  2.9863e-01, -2.4229e-01,  6.2345e-02, -5.4118e-02,
      -3.8750e-02,  4.8309e-01,  1.0859e-01, -6.5789e-01,  1.3030e-01,
      -1.1506e-01, -6.6985e-01, -5.1572e-01,  1.5001e-01, -6.2884e-01,
      -1.1048e-02, -1.6544e-01,  3.6458e-01, -5.7954e-01, -8.4940e-02,
      -2.1268e-01, -3.2703e-01, -3.1512e-01,  2.6404e-01, -8.9970e-02,
      -3.4099e-01, -5.7427e-02,  8.6417e-01, -7.0148e-02, -5.9436e-01,
      6.3583e-01, -9.9757e-01, -1.2512e+00,  6.4217e-01, -2.2527e-01,
      -1.2707e-01;
    m_params[kk].bias2 <<  0.2824, -1.1382, -0.0182,  0.6887, -0.4542, -0.9774, -1.2962,  0.7032,
      -0.0516, -0.4091, -1.2017,  0.9891,  0.1267, -0.2756, -0.2733,  0.9153;
    m_params[kk].weight4 <<  0.2954, -1.4453,  0.9166,  1.6870, -0.4972, -1.6079, -1.2639, -4.5364,
      -2.1482, -1.1003, -1.3669,  1.8386, -2.0145, -0.2177,  2.8761,  3.2093,
      -0.0727, -1.0616,  1.3330,  0.9391, -0.7901, -0.9532, -1.5739, -3.5037,
      -3.0151, -1.0467, -1.0386,  1.4425,  0.6793, -1.0830,  1.5413,  1.3248,
      -1.8036, -0.2714,  0.7623,  0.4149, -1.5170,  0.2956, -1.1731,  0.0047,
      0.0237, -0.8375, -1.2801,  0.8435,  0.0156,  0.9166, -0.0623, -0.0498;
    m_params[kk].bias4 << 1.0153, 0.9995, 0.7845;

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
