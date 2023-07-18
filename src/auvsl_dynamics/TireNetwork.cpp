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

	for(int kk = 0; kk < 4; kk++)
	{
  		m_params[kk].z_weight0 <<
			1.2566,  0.1559,  0.8143, -0.5386,  0.5636,  0.6347, -1.1558,  0.1103;
		m_params[kk].z_bias0 <<
			-2.8292,  1.9378, -0.8349, -1.6064,  1.6944,  1.5549,  2.6354, -2.0804;
		m_params[kk].z_weight2 <<
			-1.1754, -0.9655, -1.5614,  0.9377, -1.6999, -1.9281,  1.5054,  0.8243,
			1.7021,  1.1397,  1.8323, -1.6019,  1.6527,  2.1528, -1.1081, -0.6083,
			-2.3385, -0.7462, -0.5814,  0.8704, -0.6040, -0.8849,  1.2784,  0.7697,
			1.5231,  1.0196,  2.1051, -1.7811,  1.2254,  1.5881, -1.0124, -0.7982,
			1.4206,  1.3652,  2.1497, -1.1921,  1.1299,  1.7764, -0.9468, -0.6209,
			-2.2130, -0.8814,  1.6591,  0.7003, -0.9234, -0.4035,  1.2961,  1.1393,
			-1.3512, -0.9130, -2.2638,  0.7864, -0.9404, -1.1836,  1.2318,  0.9614,
			1.5407,  0.7948,  2.0888, -1.4003,  1.6644,  1.7043, -0.8409, -0.9034;
		m_params[kk].z_bias2 <<
			-0.7365,  0.6463, -0.3499,  0.9698,  0.9994, -0.5690, -0.8084,  0.8245;
		m_params[kk].z_weight4 <<
			-38.0292,  37.9480, -37.8756,  37.8352,  37.7674, -37.8351, -37.9293,
			37.8170;
		m_params[kk].z_bias4 <<
			37.8088;
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


// vx vy w zr wz
void TireNetwork::forward(const Eigen::Matrix<Scalar,8,1> &in_vec,
						  Eigen::Matrix<Scalar,num_out_features,1> &out_vec,
						  int ii)
{	
	Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> xy0_out;
	Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes,1> xy2_out;
	Eigen::Matrix<Scalar,2,1> xy4_out;

	Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes2,1> z0_out;
	Eigen::Matrix<Scalar,TireNetwork::num_hidden_nodes2,1> z2_out;
	Eigen::Matrix<Scalar,1,1> z4_out;
	Eigen::Matrix<Scalar,TireNetwork::num_in_features,1> scaled_features;
	Eigen::Matrix<Scalar,3,1> xy_features;
	Eigen::Matrix<Scalar,1,1> z_features;
	
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

	xy_features[0] = scaled_features[1];
	xy_features[1] = scaled_features[2];
	xy_features[2] = scaled_features[3];

	// Actual NN math
	xy0_out = (m_params[ii].weight0*xy_features) + m_params[ii].bias0;
	xy0_out = xy0_out.unaryExpr(&tanh_scalar_wrapper);
	xy2_out = (m_params[ii].weight2*xy0_out) + m_params[ii].bias2;
	xy2_out = xy2_out.unaryExpr(&tanh_scalar_wrapper);
	xy4_out = (m_params[ii].weight4*xy2_out) + m_params[ii].bias4;
  
	// Sign change passivity haxx
	out_vec[0] = CppAD::abs(xy4_out[0])*(1*diff);
	out_vec[1] = CppAD::abs(xy4_out[1])*(-1*in_vec[1]);	
	
	z_features[0] = scaled_features[0];

	z0_out = (m_params[ii].z_weight0*z_features) + m_params[ii].z_bias0;
	z0_out = z0_out.unaryExpr(&tanh_scalar_wrapper);
	z2_out = (m_params[ii].z_weight2*z0_out) + m_params[ii].z_bias2;
	z2_out = z2_out.unaryExpr(&tanh_scalar_wrapper);
	z4_out = (m_params[ii].z_weight4*z2_out) + m_params[ii].z_bias4;
	
	//out_vec[2] = CppAD::abs(z4_out[0])/(1 + CppAD::exp(-200*in_vec[3]));
	out_vec[2] = CppAD::abs(z4_out[0])*relu_wrapper(in_vec[3]);
	
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
			1.9183, -1.1944, -0.0941, -0.7726,  0.3715,  0.2232, -0.7302, -0.7460,
			-0.8535, -0.0809, -0.4715, -2.0354,  0.9554,  0.0775,  0.1348, -0.6217,
			0.1073,  0.3243, -0.6871,  0.3600, -0.3731, -0.3920, -0.1700,  0.0895;
		m_params[kk].bias0 <<
			-2.4556,  0.8624, -0.0141, -0.7818,  0.7625, -0.3900,  1.6314,  0.3270;
    
		m_params[kk].weight2 <<
			-8.2347e-01, -2.8110e-01,  8.4486e-01,  1.2746e-01,  5.1016e-01,
			-3.8359e-01, -8.0732e-01,  1.5990e+00,  1.6365e-01,  6.9594e-01,
			-1.8692e-01,  4.8970e-01, -5.0826e-01,  8.5592e-01,  3.0677e-01,
			9.0613e-03, -2.7446e+00, -6.1632e-01,  6.1301e-01, -5.8197e-01,
			2.7175e-01, -6.6116e-01,  2.6022e-02,  1.4282e+00, -2.8467e-03,
			3.4908e-03, -1.8357e-01,  9.9218e-01, -7.4152e-01,  8.0324e-01,
			2.4962e-01, -1.0594e-01, -5.0993e-01,  1.0599e+00,  2.3402e+00,
			5.6942e-02, -2.2344e-01, -1.1230e-03,  1.5888e+00,  3.2101e-02,
			1.5316e-01, -1.0315e+00,  6.6714e-01, -2.8029e-01,  1.1451e-01,
			-5.1009e-01, -9.4233e-01,  5.4575e-01, -6.4140e-01,  3.6725e-01,
			-9.0786e-01, -1.2710e-01, -5.3800e-01,  2.6823e-01, -4.5991e-01,
			-3.4079e-01,  4.1183e-01, -1.4952e+00, -1.2435e+00,  6.8955e-01,
			-2.4022e-01,  2.7318e-01, -5.6464e-01, -3.5930e-01;
		m_params[kk].bias2 <<
			-0.0679, -0.6706,  0.3707, -0.6988,  0.4963, -0.1490, -0.2928, -0.5737;
		
		m_params[kk].weight4 <<
			0.1867,  0.7900,  0.3217,  0.5134, -0.4663, -1.4766,  1.4382,  2.2787,
			-2.0059,  1.4770, -1.7111,  1.1529, -1.3058, -1.0913,  1.2561, -0.4580;
		m_params[kk].bias4 <<
			-0.6113, -0.9476;
	}
    
	return 0;
}





// int TireNetwork::load_model(){
// 	std::cout << "Loading Model\n";
    
// 	for(int kk = 0; kk < num_networks; kk++)
// 	{
// 		m_params[kk].weight0 <<
// 			-1.0908,  0.0159, -0.6321,  1.0150,  0.7736, -0.0288,  0.5951,  0.3307,
// 			3.6193,  0.4925,  1.4519,  0.0176, -0.7808,  1.3193, -0.2066, -1.2036,
// 			0.3647,  0.4989,  0.5400,  0.5915,  0.6415,  1.0110, -1.3339,  0.2602,
// 			1.0434,  1.3943,  1.2548, -0.1907,  0.5462, -0.0271,  1.8723, -0.8458,
// 			-0.0508, -0.2875, -0.5952,  0.7477, -0.4568, -0.3501, -0.2200, -0.8671,
// 			-0.0974, -0.3509,  0.3206,  0.0946, -0.2016,  0.7499, -0.1856,  0.1589;
// 		m_params[kk].bias0 <<
// 			-1.8866, -0.1242, -0.4053, -0.8742,  1.4843, -0.0655, -0.0974, -2.6796,
// 			0.9917, -1.0536, -2.7717, -1.0707,  0.7195,  0.9536, -0.3292, -1.3595;
    
// 		m_params[kk].weight2 <<
// 			1.8731e+00, -6.1417e-01, -1.6316e+00,  1.0204e+00,  1.3436e-01,
// 			1.4017e+00, -6.5164e-01,  8.3457e-02, -1.0361e+00,  1.6416e+00,
// 			4.9585e-01,  1.0377e+00, -4.3625e-01,  7.3021e-02,  1.3543e+00,
// 			7.1164e-02,  3.5991e-01, -1.1951e+00, -1.2058e-01,  6.2366e-01,
// 			3.7091e-01,  2.6367e-01,  1.8699e-01,  4.5941e-01, -2.8322e-01,
// 			4.1412e-01,  1.5017e+00,  1.0801e+00,  5.3708e-01, -3.3559e-01,
// 			-4.2662e-02,  5.3654e-01,  1.4809e+00, -8.7919e-01, -1.5310e+00,
// 			8.5946e-01,  2.7714e-01,  1.3090e+00, -4.0000e-01, -1.9702e-01,
// 			-8.7247e-01,  1.6473e+00,  7.4082e-01,  1.5884e+00, -6.0000e-01,
// 			1.3146e-01,  1.3704e+00,  1.7715e-01, -1.0867e-01, -1.7613e-01,
// 			-4.3189e-01, -6.1530e-02,  5.9905e-01,  4.1848e-01,  2.8376e-02,
// 			-1.5770e-02, -2.2804e-01, -5.7108e-02,  4.3507e+00,  9.0513e-01,
// 			7.2587e-01, -1.0164e+00, -5.2321e-02,  8.5920e-01,  1.5596e+00,
// 			-1.9526e-01, -1.3683e+00,  1.0255e+00, -3.8306e-02,  1.3703e+00,
// 			-4.4668e-01,  5.3449e-01, -1.2227e+00,  1.6332e+00,  4.6336e-01,
// 			8.3034e-01, -4.4501e-01,  7.3069e-02,  1.6352e+00,  3.9918e-01,
// 			2.9417e-01,  6.7598e-01, -1.3931e-01,  4.3754e-01, -5.6080e-01,
// 			-6.4664e-03, -5.0283e-01,  2.3360e+00,  1.5026e-02, -1.3735e-02,
// 			1.2131e+01, -8.0393e-02, -4.4861e-01, -8.0777e-01, -5.1042e-02,
// 			1.6111e+00,  3.9410e-01, -1.0672e+00, -1.8736e-01, -4.4771e-01,
// 			5.3983e-01,  3.4678e-01,  1.5982e-01, -1.2420e+00,  3.2306e-01,
// 			-3.2805e-02, -1.5123e+00,  1.0300e+00,  2.8789e-01,  5.4154e-01,
// 			-4.7594e-02, -3.3167e-01,  2.6083e-01,  3.3588e-01, -1.8769e-01,
// 			-7.9506e-01,  1.9358e-01,  1.3797e-01,  1.2055e-01, -1.3701e+00,
// 			-1.1434e-01,  2.4082e-01, -3.1402e+00, -8.6636e-01, -1.0771e-01,
// 			3.2321e-02, -2.8261e-01, -8.2881e-01, -2.4478e-02,  1.1828e+00,
// 			-2.8666e-01,  1.9907e-01, -3.8607e-01, -3.5480e-03,  6.9958e-01,
// 			-2.5100e-01,  4.1910e-01, -1.3478e-01,  4.4457e-01,  6.8656e-01,
// 			-6.0954e-01, -1.5904e+00, -2.7726e-01,  1.0504e-01, -3.4415e-01,
// 			-4.4473e-01,  2.9396e-01, -1.4749e-01,  3.3152e-01, -4.5791e-02,
// 			-3.8744e-01,  2.0536e-01,  2.8747e-01, -2.4065e-01, -1.2988e-02,
// 			3.9151e-01,  6.7884e-01,  4.1261e-01, -7.3090e-02, -5.7197e-01,
// 			-2.3280e-01, -8.9265e-01,  1.0550e-01, -1.9406e-01,  7.0516e-01,
// 			2.1781e-01, -3.8336e-01,  7.3525e-02,  5.0268e-02, -1.8738e-01,
// 			-7.0767e-01,  4.6155e-02,  5.5965e-01,  6.4037e-01, -4.1976e-01,
// 			-5.2703e-01,  1.2522e-02,  4.5901e-01,  4.9649e-02,  1.2853e-03,
// 			-1.3102e-01, -3.0177e-01,  7.3128e-01, -7.2562e-01,  5.9098e-01,
// 			7.1001e-02, -1.6356e+00, -5.1959e-01, -5.0424e-01, -1.2686e-01,
// 			1.5526e-01, -3.3391e-01,  1.7302e+00, -3.1199e-01, -1.3387e+00,
// 			8.4911e-01,  5.2887e-02,  1.3943e+00, -5.7890e-01, -3.4122e-01,
// 			-1.2570e+00,  1.8113e+00,  7.1960e-01,  1.2153e+00, -3.1217e-01,
// 			3.2041e-01,  1.3694e+00,  3.0871e-02,  5.6435e-01,  1.5436e-01,
// 			-5.3217e-01,  1.3711e-02,  2.7417e-01,  2.4605e-03, -1.4233e-01,
// 			-6.2438e-01, -3.3239e-01,  3.3202e-01,  7.4000e-01, -9.1985e-02,
// 			-5.7228e-01, -3.0326e-01,  3.9367e-02, -1.5580e-01,  4.1886e-01,
// 			-9.2553e-01, -4.2298e-01, -3.3919e-01,  1.5060e+00,  5.7513e-01,
// 			4.1846e-01, -2.9753e-01, -5.6015e-02, -7.2690e-02, -5.9812e-03,
// 			1.1584e+00,  3.8998e-01,  9.7119e-03, -1.9539e-01,  1.4817e-01,
// 			-2.6463e-01,  2.3671e+00,  2.4127e-01, -1.2508e-01, -1.0768e+00,
// 			-4.4289e-01,  2.7442e+00,  1.0413e+00,  1.4098e+00, -1.6042e-01,
// 			2.1561e-01,  5.4309e-01, -3.2174e-01, -3.1354e+00, -6.2581e-01,
// 			1.8214e+00;
// 		m_params[kk].bias2 <<
// 			-1.8454, -0.3805, -1.8430, -0.1643, -1.7220, -0.0798, -0.4403,  0.0474,
// 			-0.1908, -0.0535,  0.2669,  0.2776, -1.8795, -0.5584, -0.3927, -0.1593;
		
// 		m_params[kk].weight4 <<
// 			-0.3907, -1.0943, -0.4499, -0.8178, -0.5669,  0.1941, -0.8848, -0.8467,
// 			-1.1925,  0.9140,  0.4501,  0.1957, -0.3062, -0.3126, -0.7482, -1.2627,
// 			-0.9589, -0.2455, -0.8699,  1.3989, -0.8362, -1.1159,  0.1461,  0.4143,
// 			-1.1435,  0.3673,  0.4600, -1.0102, -0.8281, -0.6428, -0.4730, -1.6004;
// 		m_params[kk].bias4 <<
// 			0.4124, 0.7246;
// 	}
    
// 	return 0;
// }

