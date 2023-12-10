#include "TireNetwork.h"
#include "generated/model_constants.h"
#include <cmath>
#include <cppad/cppad.hpp>
#include <iostream>



TireNetwork::TireNetwork()
{
	out_std << 17.241097080572587, 15.9103406661298, 12.632573461434308;
	in_std_inv << 0.0008373582968488336, 0.5799984931945801, 0.576934278011322, 0.5774632096290588;
	in_std_inv = in_std_inv.cwiseInverse();
	
	for(int kk = 0; kk < 4; kk++)
	{
  		m_params[kk].z_weight0 <<
			0.4197,  0.4528,  0.4402, -0.7671, -0.4358,  0.4605,  0.4606,  0.8535;
		
		m_params[kk].z_weight2 <<
			-0.7578,  0.2568,  0.2189,  0.2418,  0.0571,  0.3125,  0.1221,  0.3023,
			-0.7722,  0.0397,  0.3921,  0.1906, -0.2118,  0.3593, -0.1007,  0.0398,
			0.8704, -0.2301,  0.2404, -0.2393, -0.1154,  0.0570, -0.2363,  0.0252,
			-0.5188,  0.4544, -0.2427,  0.0496,  0.0680,  0.1878,  0.1537,  0.1105,
			0.5801, -0.1279,  0.2275, -0.0404, -0.0301, -0.1021, -0.4477, -0.5217,
			1.0052, -0.0991, -0.1396, -0.0447,  0.2628, -0.5366, -0.3278,  0.0602,
			-0.8997,  0.0576,  0.4188,  0.2293, -0.5440,  0.0735, -0.0501,  0.2984,
			-0.5156,  0.3454,  0.2395,  0.6368, -0.0076,  0.0691, -0.0122,  0.1513;
			
		m_params[kk].z_weight4 <<
			-26.3910, -26.2104,  26.6124, -26.3831,  26.4911,  26.6735, -26.4100,
			-26.7209;	
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
	scaled_features = bekker_vec.cwiseProduct(in_std_inv);

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

	z0_out = (m_params[ii].z_weight0*z_features);
	z0_out = z0_out.unaryExpr(&tanh_scalar_wrapper);
	z2_out = (m_params[ii].z_weight2*z0_out);
	z2_out = z2_out.unaryExpr(&tanh_scalar_wrapper);
	z4_out = (m_params[ii].z_weight4*z2_out);
	
	//forces[2] = CppAD::abs(z4_out[0])*relu_wrapper(in_vec[3]);
	forces[2] = relu_wrapper(z4_out[0]);
	
	// Scale output
	forces = forces.cwiseProduct(out_std);
	
	out_vec[0] = forces[0];
	out_vec[1] = forces[1];
	out_vec[2] = forces[2];
	out_vec[3] = 0;
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
			1.8523e+00, -2.4395e-01, -2.0577e+00,  5.7221e-01,  9.7006e-01,
			3.7060e-01, -6.8117e-02,  1.5170e+00,  6.9176e-01, -1.5698e-01,
			-1.1519e+00,  4.7048e-01, -1.0036e-02,  1.4627e-01,  1.8264e+00,
			-1.3942e+00,  7.5852e-02,  5.7577e-03,  1.8568e+00, -3.1544e-01,
			2.0023e+00, -1.6360e+01, -1.1013e+00,  2.8912e-02;
		
		m_params[kk].weight2 <<
			-0.6545,  1.2153,  0.2281, -0.8833, -1.4218, -2.7572,  0.3027, -2.1109,
			-2.0394, -0.0442,  0.3384,  0.2103, -1.5120,  1.2095,  0.1672,  0.1592,
			0.5422,  0.3246, -0.0478,  0.3761,  0.7557, -1.2539,  0.8754, -0.4485,
			-1.3457, -1.3106, -0.0682,  1.4183, -0.9145,  4.1784, -0.4422,  1.5920,
			-0.4213,  0.2675, -0.1125,  0.0500,  1.3408,  1.1443,  0.3911,  1.5622,
			-0.6181, -0.3097,  0.4885,  0.4184,  0.8413,  0.4065,  0.7734,  0.3717,
			0.2096, -1.8048, -0.1093, -1.7614, -0.7847,  0.1165, -0.5894,  0.0641,
			0.0843,  1.7857, -0.8906, -1.0495,  0.7428, -2.3685, -0.3676, -1.9639;
		
		m_params[kk].weight4 <<
			0.3660, -0.2972,  0.5233,  1.0919, -0.2640, -0.1887, -0.1882,  0.6705,
			-2.1059, -0.4484, -0.3344, -0.9591,  0.3450, -0.7008,  0.4277,  1.2480;
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
