#include <vector>
#include "gtest/gtest.h"

#include <matplotlibcpp.h>
#include <cpp_bptt.h>
#include <cpp_neural.h>

#include "HybridDynamics.h"
#include "VehicleSystem.h"
#include "utils.h"

namespace plt = matplotlibcpp;

namespace {

	class VehicleFixture : public ::testing::Test
	{
	public:
		std::vector<VectorF> m_gt_list;
		std::vector<VectorAD> m_gt_list_adf;
		VectorAD m_x0;
		VectorAD m_params;

		std::shared_ptr<VehicleSystem<ADF>>    m_system_adf;

		bool loadVec(VectorAD &params, const std::string &file_name)
		{
			char comma;
			std::ifstream data_file(file_name);
			if(!data_file.is_open())
			{
				std::cout << "Failed to open file\n";
				return true;
			}
	  
			for(int i = 0; i < params.size(); i++)
			{
				data_file >> params[i];
				data_file >> comma;
			}
	  
			return false;
		}  
    
		VehicleFixture()
		{
			srand(time(NULL)); // randomize seed
      
			m_system_adf    = std::make_shared<VehicleSystem<ADF>>();
      
			m_params = VectorAD::Zero(m_system_adf->getNumParams());
			m_x0 = VectorAD::Zero(m_system_adf->getStateDim());
			
			m_system_adf->getDefaultInitialState(m_x0);
			m_x0[6] = .0605;

			loadVec(m_params, "/home/justin/tire.net");
			m_system_adf->setParams(m_params);
			
			m_gt_list.resize(m_system_adf->getNumSteps());
			m_gt_list_adf.resize(m_system_adf->getNumSteps());
      
			for(int i = 0; i < m_gt_list.size(); i++)
			{
				m_gt_list[i] = VectorF::Zero(m_system_adf->getStateDim());
				m_gt_list_adf[i] = VectorAD::Random(m_system_adf->getStateDim());
				for(int j = 0; j < m_gt_list_adf[i].size(); j++)
				{
					m_gt_list[i][j] = CppAD::Value(m_gt_list_adf[i][j]);
				}
			}
		}
		~VehicleFixture(){}
    
		VectorF getGradientSimple()
		{      
			CppAD::Independent(m_params);
			m_system_adf->setParams(m_params);
      
			std::vector<VectorAD> x_list(m_system_adf->getNumSteps());
      
			VectorAD loss(1);
      
			loss[0] = 0;
			for(int i = 0; i < x_list.size(); i++)
			{
				loss[0] += m_system_adf->loss(m_gt_list_adf[i], x_list[i]);
			}
      
			CppAD::ADFun<double> func(m_params, loss);
      
			VectorF y0(1);
			y0[0] = 1;

			std::cout << "Loss " << CppAD::Value(loss[0]) << "\n";
      
			return func.Reverse(1, y0);
		}
	};

  


  
	// TEST_F(VehicleFixture, validate_gradient_easy)
	// {
	// 	VectorF grad_simple = getGradientSimple();
	// 	//VectorF grad_bptt = getGradientBPTT();

	// 	for(int i = 0; i < m_system_adf->getNumParams(); i++)
	// 	{
      
	// 	}
	// }
  
  
	TEST_F(VehicleFixture, settling)
	{
		int num_steps = 100;
		std::vector<double> time(num_steps);
		std::vector<double> elev(num_steps);
		std::vector<double> x_vec(num_steps);
		std::vector<double> y_vec(num_steps);
    
		m_system_adf->setParams(m_params);

		VectorAD xk(m_system_adf->getStateDim());
		VectorAD xk1(m_system_adf->getStateDim());
    
		xk = m_x0;
    
		for(int i = 0; i < num_steps; i++)
		{
			xk[HybridDynamics::STATE_DIM] = 0.0; //vl
			xk[HybridDynamics::STATE_DIM+1] = 0.0; //vr
			m_system_adf->integrate(xk, xk1);
      
			time[i] = i * CppAD::Value(HybridDynamics::timestep);
			elev[i] = CppAD::Value(xk1[6]);
			x_vec[i] = CppAD::Value(xk1[4]);
			y_vec[i] = CppAD::Value(xk1[5]);
			xk = xk1;
		}
    
		plt::subplot(1,2,1);
		plt::title("X-Y plot");
		plt::plot(x_vec, y_vec);
    
		plt::subplot(1,2,2);
		plt::title("Time vs Elevation");
		plt::plot(time, elev);
    
		plt::show();
	}
  
	TEST_F(VehicleFixture, no_lateral_drift)
	{
		int num_steps = 1000;
		std::vector<double> time(num_steps);
		std::vector<double> elev(num_steps);
		std::vector<double> x_vec(num_steps);
		std::vector<double> y_vec(num_steps);
		
		m_system_adf->setParams(m_params);
		
		VectorAD xk(m_system_adf->getStateDim());
		VectorAD xk1(m_system_adf->getStateDim());
		
		xk = m_x0;
		xk[15] = .1; //add an initial lateral movement
		
		for(int i = 0; i < num_steps; i++)
		{
			xk[HybridDynamics::STATE_DIM] = 0.0; //vl
			xk[HybridDynamics::STATE_DIM+1] = 0.0; //vr
			m_system_adf->integrate(xk, xk1);
			
			time[i] = i * 0.01;
			elev[i] = CppAD::Value(xk1[6]);
			x_vec[i] = CppAD::Value(xk1[4]);
			y_vec[i] = CppAD::Value(xk1[5]);
			xk = xk1;
		}
    
		plt::subplot(1,2,1);
		plt::title("X-Y plot");
		plt::xlabel("[m]");
		plt::ylabel("[m]");
		plt::plot(x_vec, y_vec, "-bo");
    
		plt::subplot(1,2,2);
		plt::title("Time vs Elevation");
		plt::xlabel("Time [s]");
		plt::ylabel("Elevation [m]");
		plt::plot(time, elev);
    
		plt::show();

		// Model should drift between this range.
		EXPECT_GT(y_vec.back(), 0.0);
		EXPECT_LT(y_vec.back(), 0.1);
	}

	
	TEST_F(VehicleFixture, straight)
	{
		int num_steps = 1000;
		std::vector<double> time(num_steps);
		std::vector<double> elev(num_steps);
		std::vector<double> x_vec(num_steps);
		std::vector<double> y_vec(num_steps);
    
		m_system_adf->setParams(m_params);

		VectorAD xk(m_system_adf->getStateDim());
		VectorAD xk1(m_system_adf->getStateDim());
    
		xk = m_x0;
    
		for(int i = 0; i < num_steps; i++)
		{
			xk[HybridDynamics::STATE_DIM] = 4; //vl
			xk[HybridDynamics::STATE_DIM+1] = 4; //vr
			m_system_adf->integrate(xk, xk1);
      
			time[i] = i * 0.01;
			elev[i] = CppAD::Value(xk1[6]);
			x_vec[i] = CppAD::Value(xk1[4]);
			y_vec[i] = CppAD::Value(xk1[5]);
			xk = xk1;
		}
    
		plt::subplot(1,2,1);
		plt::title("X-Y plot");
		plt::xlabel("[m]");
		plt::ylabel("[m]");
		plt::plot(x_vec, y_vec);
    
		plt::subplot(1,2,2);
		plt::title("Time vs Elevation");
		plt::xlabel("Time [s]");
		plt::ylabel("Elevation [m]");
		plt::plot(time, elev);
    
		plt::show();

		// tire_radius is .098m * 1rad/s
		std::cout << "Distance: " << x_vec.back() << "\n";
		EXPECT_NEAR(x_vec.back(), 10*.098, 5e-2);
	}

	TEST_F(VehicleFixture, circle)
	{
		int num_steps = 10000;
		std::vector<double> time(num_steps);
		std::vector<double> elev(num_steps);
		std::vector<double> x_vec(num_steps);
		std::vector<double> y_vec(num_steps);
    
		m_system_adf->setParams(m_params);

		VectorAD xk(m_system_adf->getStateDim());
		VectorAD xk1(m_system_adf->getStateDim());
    
		xk = m_x0;

		double max_x = CppAD::Value(xk[4]);
		double min_x = CppAD::Value(xk[4]);
		double max_y = CppAD::Value(xk[5]);
		double min_y = CppAD::Value(xk[5]);
    
		for(int i = 0; i < num_steps; i++)
		{
			xk[HybridDynamics::STATE_DIM] = 2; //vl
			xk[HybridDynamics::STATE_DIM+1] = 1; //vr
			m_system_adf->integrate(xk, xk1);
      
			time[i] = i * 0.01;
			elev[i] = CppAD::Value(xk1[6]);
			x_vec[i] = CppAD::Value(xk1[4]);
			y_vec[i] = CppAD::Value(xk1[5]);
			xk = xk1;

			if(x_vec[i] > max_x)
			{
				max_x = x_vec[i];
			}
			if(x_vec[i] < min_x)
			{
				min_x = x_vec[i];
			}
      
			if(y_vec[i] > max_y)
			{
				max_y = y_vec[i];
			}
			if(y_vec[i] < min_y)
			{
				min_y = y_vec[i];
			}
		}
    
		plt::subplot(1,2,1);
		plt::title("X-Y plot");
		plt::xlabel("[m]");
		plt::ylabel("[m]");
		plt::plot(x_vec, y_vec);
    
		plt::subplot(1,2,2);
		plt::title("Time vs Elevation");
		plt::xlabel("Time [s]");
		plt::ylabel("Elevation [m]");
		plt::plot(time, elev);
    
		plt::show();

		EXPECT_NEAR((max_x - min_x), (max_y - min_y), 1e-2);
	}

	
	// Test to see with an initial 10rad/s wz (yaw rate)
	// Tire-soil model deactivated
	// Expected behavior:
	// No linear velocity except in vz (due to falling)
	// Roughly same wz (Will be slightly different because wz is not expressed at the center of the mass of the vehicle, and there will be some angular precession)
	// Still fails with no gravity and no external forces. Fuck.
	TEST_F(VehicleFixture, test_wz)
	{
		int num_steps = 100;
		std::vector<double> time(num_steps);
		std::vector<double> yaw_rate(num_steps);
		std::vector<double> x_vec(num_steps);
		std::vector<double> y_vec(num_steps);

		// This effectively deactivates the tire-soil network
		// So no external forces will be acting on the vehicle.
		m_params = VectorAD::Zero(m_system_adf->getNumParams());
		m_system_adf->setParams(m_params);
		
		VectorAD xk(m_system_adf->getStateDim());
		VectorAD xk1(m_system_adf->getStateDim());

		constexpr double wz_i = 10.0;
		Scalar com_state[m_system_adf->m_hybrid_dynamics.STATE_DIM] = {0,0,0,1, 0,0,100, 0,0,0,0, 0,0,wz_i, 0,0,0, 0,0,0,0};
		Scalar base_state[m_system_adf->m_hybrid_dynamics.STATE_DIM];
		m_system_adf->m_hybrid_dynamics.initStateCOM(com_state, base_state);
		
		for(int i = 0; i < m_system_adf->m_hybrid_dynamics.STATE_DIM; i++)
		{
			xk[i] = base_state[i];
		}
		
		double max_x = CppAD::Value(xk[4]);
		double min_x = CppAD::Value(xk[4]);
		double max_y = CppAD::Value(xk[5]);
		double min_y = CppAD::Value(xk[5]);
		
		for(int i = 0; i < num_steps; i++)
		{
			xk[11] = base_state[11];
			xk[12] = base_state[12];
			xk[13] = base_state[13];
			xk[14] = base_state[14];
			xk[15] = base_state[15];
			xk[16] = base_state[16];
			
			xk[HybridDynamics::STATE_DIM] = 0; //vl
			xk[HybridDynamics::STATE_DIM+1] = 0; //vr
			m_system_adf->integrate(xk, xk1);
			
			time[i] = i * 0.01;
			yaw_rate[i] = CppAD::Value(xk1[13]);
			x_vec[i] = CppAD::Value(xk1[4]);
			y_vec[i] = CppAD::Value(xk1[5]);
			xk = xk1;
			
			if(x_vec[i] > max_x)
			{
				max_x = x_vec[i];
			}
			if(x_vec[i] < min_x)
			{
				min_x = x_vec[i];
			}
      
			if(y_vec[i] > max_y)
			{
				max_y = y_vec[i];
			}
			if(y_vec[i] < min_y)
			{
				min_y = y_vec[i];
			}
		}
		
		plt::subplot(1,2,1);
		plt::title("X-Y plot");
		plt::xlabel("[m]");
		plt::ylabel("[m]");
		plt::plot(x_vec, y_vec);
		
		plt::subplot(1,2,2);
		plt::title("Time vs Yaw Rate");
		plt::xlabel("Time [s]");
		plt::ylabel("Yaw Rate [rad/s]");
		plt::plot(time, yaw_rate);
		
		plt::show();
		
		Scalar roll, pitch, yaw;
		toEulerAngles(xk1[3], xk1[0], xk1[1], xk1[2],
					  roll, pitch, yaw);

		std::cout << "Quaternion: "
				  << CppAD::Value(xk1[0]) << ", "
				  << CppAD::Value(xk1[1]) << ", "
				  << CppAD::Value(xk1[2]) << ", "
				  << CppAD::Value(xk1[3]) << "\n";
		
		Scalar final_yaw = wz_i*num_steps*.01;
		std::cout << CppAD::Value(final_yaw) << std::endl;
		final_yaw = CppAD::atan2(CppAD::sin(final_yaw), CppAD::cos(final_yaw));
		
		EXPECT_NEAR(CppAD::Value(yaw), CppAD::Value(final_yaw), 1e-3);
	}

	TEST_F(VehicleFixture, angle_conversion)
	{
		Scalar roll, pitch, yaw;
		toEulerAngles(-0.979752, 0.0661528, 0.0231171, 0.187553,
					  roll, pitch, yaw);
		std::cout << CppAD::Value(roll) << ", "
				  << CppAD::Value(pitch) << ", "
				  << CppAD::Value(yaw) << "\n";
			
	}
}
