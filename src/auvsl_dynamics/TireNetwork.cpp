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
	Eigen::Matrix<Scalar,3,1> forces;
	
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
	
	// Apply scaling after calculating the bekker features from kinematics
	scaled_features = (bekker_vec - in_mean).cwiseProduct(in_std_inv);

	xy_features[0] = scaled_features[1];
	xy_features[1] = scaled_features[2];
	xy_features[2] = scaled_features[3];

	// Actual NN math
	xy0_out = (m_params[ii].weight0*xy_features);
	xy0_out = xy0_out.unaryExpr(&tanh_scalar_wrapper);
	xy2_out = (m_params[ii].weight2*xy0_out);
	xy2_out = xy2_out.unaryExpr(&tanh_scalar_wrapper);
	xy4_out = (m_params[ii].weight4*xy2_out);
	
	forces[0] = xy4_out[0];
	forces[1] = xy4_out[1];
	
	z_features[0] = scaled_features[0];

	z0_out = (m_params[ii].z_weight0*z_features) + m_params[ii].z_bias0;
	z0_out = z0_out.unaryExpr(&tanh_scalar_wrapper);
	z2_out = (m_params[ii].z_weight2*z0_out) + m_params[ii].z_bias2;
	z2_out = z2_out.unaryExpr(&tanh_scalar_wrapper);
	z4_out = (m_params[ii].z_weight4*z2_out) + m_params[ii].z_bias4;
	
	forces[2] = CppAD::abs(z4_out[0])*relu_wrapper(in_vec[3]);
	
	// Scale output
	forces = forces.cwiseProduct(out_std);
	
	// L1 to calculate gating value
	Scalar gate = CppAD::tanh(CppAD::abs(xy_features[0]) +
							  CppAD::abs(xy_features[1]) +
							  CppAD::abs(xy_features[2])
							  );
	
	out_vec[0] = forces[0];
	out_vec[1] = forces[1];
	out_vec[2] = forces[2];
	out_vec[3] = relu_wrapper(forces[0]*-diff) +
		         relu_wrapper(forces[1]*in_vec[1]); // Penalty
}


int TireNetwork::getNumParams()
{
  return 4*(m_params[0].weight0.size() +
			m_params[0].weight2.size() +
			m_params[0].weight4.size()
			);
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
		
		for(int i = 0; i < m_params[kk].weight2.rows(); i++)
		{
			for(int j = 0; j < m_params[kk].weight2.cols(); j++)
			{
				m_params[kk].weight2(i,j) = params[idx];
				idx++;
			}
		}

		for(int i = 0; i < m_params[kk].weight4.rows(); i++)
		{
			for(int j = 0; j < m_params[kk].weight4.cols(); j++)
			{
				m_params[kk].weight4(i,j) = params[idx];
				idx++;
			}
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
  
		for(int i = 0; i < m_params[kk].weight2.rows(); i++)
		{
			for(int j = 0; j < m_params[kk].weight2.cols(); j++)
			{
				params[idx] = m_params[kk].weight2(i,j);
				idx++;
			}
		}

		for(int i = 0; i < m_params[kk].weight4.rows(); i++)
		{
			for(int j = 0; j < m_params[kk].weight4.cols(); j++)
			{
				params[idx] = m_params[kk].weight4(i,j);
				idx++;
			}
		}
	}
}



int TireNetwork::load_model(){
	std::cout << "Loading Model\n";
    
	for(int kk = 0; kk < num_networks; kk++)
	{
		m_params[kk].weight0 <<
			1.3236, -0.0296,  1.0797, -0.5106, -0.2236, -2.7231,  0.9010,  0.0896,
			-2.1982, -0.4364, -1.3233, -0.4495,  9.8746,  0.6139,  0.0833, -1.2986,
			0.1669,  0.7433, -9.2812, -1.8801, -0.4478,  0.9916,  0.9795, -0.4905;
		
		m_params[kk].weight2 <<
			-1.3760, -0.7857,  1.0379,  0.8910, -3.6161,  1.5966,  0.1276, -1.7902,
			0.8242, -0.1313,  0.8048, -0.0164,  3.7134, -1.2779, -1.3999,  0.9186,
			-1.7484,  1.4101, -1.9800, -0.0733,  0.1897,  1.2107,  0.1620, -0.0420,
			-0.6605,  0.3101, -0.0374, -1.6281, -2.8861,  1.1062,  2.4163, -1.2547,
			0.5712,  0.1898, -1.1601, -0.1508,  0.2089,  0.1852, -0.4988,  1.1023,
			1.1551,  1.4035,  0.5221,  0.3638,  1.0377, -0.4549,  0.4757,  0.3139,
			-0.6602,  3.0073,  0.1186, -0.0431, -1.7339,  3.0759, -0.7352,  1.2115,
			-1.0978,  0.5640,  1.0522, -0.0414, -0.5647,  0.4006, -0.4100,  0.6935;
		m_params[kk].weight4 <<
			1.1821, -0.0664, -0.5187, -1.2452,  0.1416,  0.5036, -0.2592, -0.1049,
			-0.3164, -1.1725,  0.0809, -0.9028, -0.2043,  0.7160,  0.2720,  0.5577;
	}
    
	return 0;
}


/*
int TireNetwork::load_model(){
	std::cout << "Loading Model\n";
    
	for(int kk = 0; kk < num_networks; kk++)
	{
		m_params[kk].weight0 <<
			-0.2468, -0.1325,  1.0403, 13.0290,  2.1225,  0.1050, -1.2154, -1.7481,
			-1.7024,  0.9309,  2.9887, -0.4881, -0.8420, -3.2459, -0.4109,  7.8091,
			0.6007,  0.0927,  0.4663, -0.0444,  0.7972,  2.0333,  1.7781, -1.5263,
			1.5861,  0.5968,  4.3453,  0.1087, -0.8261,  5.2026, -2.4750,  0.2200,
			10.7359, 26.8596,  0.7313,  0.1003, -0.7321,  0.0611,  0.1137, -4.3404,
			0.2183,  0.2456, -0.4701,  0.6623,  1.3374, -3.3448,  0.1627, -1.3036;		
		m_params[kk].weight2 <<
			8.3206e-01,  1.3539e+00, -4.6304e-01, -2.5791e-01, -3.8296e-01,
			-1.5562e+00, -8.0693e-01,  4.5575e-04,  7.3248e-01, -6.7096e-02,
			2.7747e-01, -9.1181e-01,  1.2638e+00,  1.2748e-01,  5.7327e-01,
			3.5803e-01,  1.4984e+00, -7.0646e-01, -1.1692e-01, -1.3286e-02,
			7.9611e-02,  6.1109e-01,  6.1121e-01,  1.0117e-01,  7.0678e-02,
			2.5787e-01,  9.2770e-01,  7.0797e-01, -1.2662e+00, -6.2745e-01,
			8.5975e-02, -2.4767e-01, -1.6251e+00,  3.3053e-01,  8.3474e-02,
			-2.5503e-02, -6.3748e-02, -6.7216e-04, -4.5521e-01, -2.4421e-01,
			-1.2687e-01, -3.3974e-01, -9.6771e-01, -1.2744e+00,  1.1726e+00,
			7.3186e-01,  8.1371e-02,  1.5446e-01,  1.3012e+00,  9.7184e-01,
			2.1296e-02,  2.2177e-01,  2.6590e-01, -2.4421e-01,  1.1813e-02,
			-1.9184e-02,  9.4203e-01,  9.0794e-01,  3.4590e-01, -1.4082e+00,
			1.5206e+00,  8.3654e-01,  5.0386e-01,  8.8715e-01, -1.5868e+00,
			7.7619e-01,  8.0396e-01,  3.7056e-01,  4.9049e-01,  3.6583e+00,
			1.0677e-02, -8.4868e-01, -8.2891e-01,  5.8937e-01,  9.1169e-01,
			2.2220e-01, -1.2031e+00, -8.9039e-02,  3.9519e-02, -1.5902e+00,
			7.8942e-02,  1.6492e+00, -7.0304e-02, -3.9819e-01,  9.2745e-01,
			4.6972e+00,  2.7175e-01, -7.2802e-01, -1.0440e-01,  5.4466e-01,
			-6.3282e-01, -6.0481e-01, -7.6338e-01, -3.7518e-01, -8.8698e-01,
			-3.0805e-01,  1.1539e-01, -4.2761e+00, -7.9126e-03, -4.1042e-01,
			8.8047e-01, -4.2450e+00, -2.6609e-01, -3.0181e-01,  3.4025e-01,
			-2.7111e-01,  3.2581e-03, -2.5044e+00,  1.8786e+00,  1.7469e+00,
			3.9359e-01, -4.2744e-02,  1.1176e+00, -5.3274e+00,  1.6531e-01,
			-1.5986e+00,  4.4369e-01, -3.8034e+00,  1.6629e-01,  7.6300e-02,
			3.4443e-01, -4.5227e-01, -7.2820e-02, -3.5612e+00,  2.4329e+00,
			2.4379e+00, -2.8795e-01,  1.0163e+00,  1.0162e+00, -1.1212e+00,
			-3.1332e-01, -1.5229e+00, -1.3998e+00,  1.9671e-01, -4.9209e-01,
			-3.2685e-01,  2.5858e-01,  2.5084e-01, -3.2933e-02,  1.6091e-01,
			-1.0485e+00,  1.8550e-01,  6.0978e-01, -3.3629e-01,  6.8141e-01,
			-5.2878e+00,  2.7841e-01, -2.4749e-01,  1.1750e-01, -3.2658e+00,
			-8.2657e-01, -4.4563e-01,  2.7555e-01, -3.1147e-01, -2.9064e-03,
			-2.3342e+00,  3.2423e+00,  2.3367e+00,  2.0445e-01,  2.9517e-01,
			-5.4627e-01,  6.4559e-01, -1.8579e-01, -3.8677e-01,  1.1937e-02,
			8.8715e-01,  6.6189e-01,  1.7234e-01, -3.5152e-01, -1.9206e-01,
			1.1790e-02, -3.9081e-01, -9.2241e-01,  2.0859e-01, -7.2060e-01,
			-3.8127e-01,  7.1931e-01, -9.5658e-01, -3.2307e-03,  5.4941e-01,
			6.4207e-02, -9.8093e-01, -9.1448e-01, -2.3280e-01, -5.1083e-01,
			4.7734e-01,  2.7683e-01,  5.4167e-01,  4.7266e-01, -2.6062e-01,
			1.7731e-01, -4.5544e-01,  9.0517e-01,  5.6788e-01,  4.0924e-02,
			2.2362e-01,  6.1825e-02, -7.1823e-01, -1.9000e+00,  5.9676e-01,
			-5.0229e-01, -9.2803e-01,  2.8031e-01, -6.6147e-01,  1.4668e+00,
			5.9305e-01, -1.6981e-01, -6.5861e-02, -9.1447e-01, -1.6571e-01,
			3.6240e-01,  1.3944e+00,  1.2744e+00,  5.3418e-01,  9.5235e-02,
			3.2751e-01, -3.8961e-01, -1.9762e-01,  5.5813e-02, -7.4070e-02,
			4.7859e-01, -6.7751e-02, -5.2880e-01, -2.4283e-01, -3.0543e+00,
			1.4647e-01,  3.7114e-02, -3.9444e-01, -2.7180e-01, -2.3080e-01,
			1.8783e+00,  6.7980e-02,  1.1598e+00, -8.4439e-01, -2.9366e-01,
			-7.5524e-01, -1.2884e+00, -5.4797e-01,  5.3815e-01, -4.0095e-01,
			-1.7269e+00,  1.4038e+00,  6.1499e-01,  3.4848e-01,  1.2126e+00,
			3.7202e+00, -1.8503e-01,  6.6655e-01,  1.5267e+00, -2.4038e-01,
			-6.6244e-01, -7.5793e-01, -1.9019e+00,  7.1368e-01,  7.3259e-01,
			-2.1852e+00;			
		m_params[kk].weight4 <<
			-0.2370, -0.1881, -0.4961, -0.1917, -0.1074,  0.1995,  0.7244,  0.3250,
			-0.5822, -1.0941,  0.5761,  0.4736, -0.2936, -0.5561,  0.1654, -0.0625,
			-0.2990, -1.0474, -0.6244, -0.2798, -0.3429,  0.1827,  1.2253, -1.1353,
			0.5760, -0.0151, -0.6773, -0.5517,  0.2610,  0.9495, -0.0021,  0.2358;
	}
    
	return 0;
}
*/
