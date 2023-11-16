#include <matplotlibcpp.h>

#include "gtest/gtest.h"
#include "TestTerrainMaps.h"
#include "BekkerTireModel.h"
#include "BekkerSystem.h"
#include "generated/model_constants.h"
#include "types/Scalars.h"

#include <iostream>
#include <fenv.h>
#include <sstream>

namespace plt = matplotlibcpp;

namespace{
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, 1> VectorAD;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> MatrixAD;
	
	class BekkerFixture : public ::testing::Test
	{
	public:
		VectorAD m_x0;
		VectorAD m_params;

		std::shared_ptr<BekkerSystem<ADF>>    m_system_adf;

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

		void plot_cross(float x1, float x2, float y1, float y2)
		{
			std::vector<float> x(2);
			std::vector<float> y(2);

			x[0] = x1;
			x[1] = x2;
			y[0] = 0;
			y[1] = 0;
			plt::plot(x, y, "b");
			x[0] = 0;
			x[1] = 0;
			y[0] = y1;
			y[1] = y2;
			plt::plot(x, y, "b");    
		}

		
		BekkerFixture()
		{
			//feenableexcept(FE_INVALID | FE_OVERFLOW);
			srand(time(NULL)); // randomize seed

			auto map = std::make_shared<const FlatTerrainMap<ADF>>();
			m_system_adf = std::make_shared<BekkerSystem<ADF>>(map);
      
			m_params = VectorAD::Zero(m_system_adf->getNumParams());
			m_x0 = VectorAD::Zero(m_system_adf->getStateDim() + m_system_adf->getControlDim());
			
			loadVec(m_params, "/home/justin/tire.net");
			m_system_adf->getDefaultParams(m_params);
			m_system_adf->setParams(m_params);

			m_system_adf->getDefaultInitialState(m_x0);
			m_x0[6] = .0605;

			ADF z_stable = m_x0[6];
			std::cout << "Stable Quaternion: "
					  << m_x0[0] << ", "
					  << m_x0[1] << ", "
					  << m_x0[2] << ", "
					  << m_x0[3] << "\n";
		
			std::cout << "z_stable: " << z_stable << "\n";

		}
	};

	// The purpose of this test is to check that tau functions in BekkerTireModel are smooth and continuous
	// Because I had to hack in an if statement to avoid a divide by zero.
	TEST_F(BekkerFixture, tau)
	{
		BekkerTireModel tire_model;
	  
		Eigen::Matrix<ADF,8,1> features;
	  
		features[0] = 0.004;
		features[1] = 0.1;
		features[2] = 0.1;
	  
		features[3] = 29.76;
		features[4] = 2083.0;
		features[5] = 0.8;
		features[6] = 0.0;
		features[7] = 0.3927;
		
		tire_model.get_forces(features);
		
		int len = 100;
		std::vector<float> theta_vec(len);
		std::vector<float> tau_x_cf_vec(len);
		std::vector<float> tau_x_cc_vec(len);
		std::vector<float> tau_x_cr_vec(len);
		
		for(int i = 0; i < len; i++)
		{
			ADF theta = 1e-1*ADF((2.0*i/(float)len) - 1.0);
			
			theta_vec[i] = CppAD::Value(theta);
			tau_x_cf_vec[i] = CppAD::Value(tire_model.tau_x_cf(theta));
			tau_x_cc_vec[i] = CppAD::Value(tire_model.tau_x_cc(theta));
			tau_x_cr_vec[i] = CppAD::Value(tire_model.tau_x_cr(theta));
		}
		
		plot_cross(-1.0, 1.0, -1.0, 1.0);
		plt::plot(theta_vec, tau_x_cf_vec, {{"label","tau_x_cf"}});
		plt::plot(theta_vec, tau_x_cc_vec, {{"label","tau_x_cc"}});
		plt::plot(theta_vec, tau_x_cr_vec, {{"label","tau_x_cr"}});
		plt::legend();
		
		plt::title("Theta vs Tau");
		plt::show();		
	}


	// Again with the divide by zero
	// Slip ratio sucks.
	TEST_F(BekkerFixture, slip_ratio)
	{
		BekkerTireModel tire_model;
		
		Eigen::Matrix<ADF,4,1> inputs;
		Eigen::Matrix<ADF,3,1> bekker_features;
		
		int len = 100;
		std::vector<float> vx_vec(len);
		std::vector<float> slip_ratio_vec(len);
		
		for(int j = 0; j < 10; j++)
		{
			ADF tire_vx = j *0.1;
			for(int i = 0; i < len; i++)
			{
				ADF tangent_vx = 1.0*ADF((2.0*i/(float)len) - 1.0);
				
				inputs[0] = 0.004;
				inputs[1] = tire_vx;
				inputs[2] = 0.0;
				inputs[3] = tangent_vx;

				bekker_features = tire_model.get_features(inputs);
				
				vx_vec[i] = CppAD::Value(tangent_vx);
				slip_ratio_vec[i] = CppAD::Value(bekker_features[1]);
			}
		
			plt::plot(vx_vec, slip_ratio_vec);
		}
		
		plot_cross(-1.0, 1.0, -1.0, 1.0);
		plt::title("Tangent Velocity vs Slip Ratio");
		plt::show();		
	}
	
	
	TEST_F(BekkerFixture, vx_fx_plot)
	{
		BekkerTireModel tire_model;
		
		Eigen::Matrix<ADF,8,1> features;
		Eigen::Matrix<ADF,4,1> inputs;
		Eigen::Matrix<ADF,3,1> bekker_features;
		Eigen::Matrix<ADF,4,1> forces;
	  
		features[0] = 0;
		features[1] = 0;
		features[2] = 0;
	  
		features[3] = 29.758547;
		features[4] = 2083.0;
		features[5] = 1.197933;
		features[6] = 0.102483;
		features[7] = 0.652405;
    
		int len = 1000;
		int num = 11;
		std::vector<float> vx_vec(len);
		std::vector<float> fx_vec(len);
		
		ADF tire_vy = 0;


		for(int j = 0; j < num; j++)
		{
			ADF tire_vx = 0.1*ADF((2.0*j/(float)(num - 1)) - 1.0);
			
			for(int i = 0; i < len; i++)
			{
				ADF tangent_vx = 0.2*ADF((2.0*i/(float)len) - 1.0); // + tire_vx;
				
				inputs[0] = 0.004; //zr
				inputs[1] = tire_vx;
				inputs[2] = 0.0;
				inputs[3] = tangent_vx;
				
				bekker_features = tire_model.get_features(inputs);
				
				features[0] = bekker_features[0];
				features[1] = bekker_features[1];
				features[2] = bekker_features[2];
				
				forces = tire_model.get_forces(features);
				
				forces[0] = CppAD::CondExpGt(tangent_vx - tire_vx, ADF(0.0), CppAD::abs(forces[0]), -CppAD::abs(forces[0]));
				
				vx_vec[i] = CppAD::Value(tangent_vx - tire_vx);
				fx_vec[i] = CppAD::Value(forces[0]);
			}
			
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << CppAD::Value(tire_vx);
			std::string label = stream.str();
			
			float grey = (0.8*j / num);
			plt::plot(vx_vec, fx_vec, {{"color", std::to_string(grey)}, {"label", label}});
		}
		
		plt::legend();
		plt::xlabel("Velocity Difference (m/s)");
		plt::ylabel("Longitudinal Force (N)");
		plt::show();
	}




	TEST_F(BekkerFixture, vy_fy_plot)
	{
		BekkerTireModel tire_model;
	  
		Eigen::Matrix<ADF,8,1> features;
		Eigen::Matrix<ADF,4,1> forces;
		Eigen::Matrix<ADF,4,1> inputs;
		Eigen::Matrix<ADF,3,1> bekker_features;
		
		features[0] = 0;
		features[1] = 0;
		features[2] = 0;
	  
		features[3] = 29.758547;
		features[4] = 2083.0;
		features[5] = 1.197933;
		features[6] = 0.102483;
		features[7] = 0.652405;

		int num = 11;
		int len = 10000;
		std::vector<float> vy_vec(len);
		std::vector<float> fy_vec(len);

		ADF tire_tangent_vel = 0.1;
	  
		for(int j = 0; j < num; j++)
		{
			ADF tire_vx = (float)j/(num-1);
			for(int i = 0; i < len; i++)
			{
				ADF tire_vy = 1*ADF((2.0*i/(float)len) - 1.0);
			  
				inputs[0] = 0.001; //zr
				inputs[1] = tire_vx;
				inputs[2] = tire_vy;
				inputs[3] = tire_tangent_vel;
				
				bekker_features = tire_model.get_features(inputs);
				
				features[0] = bekker_features[0];
				features[1] = bekker_features[1];
				features[2] = bekker_features[2];
				
				forces = tire_model.get_forces(features);
				
				vy_vec[i] = CppAD::Value(tire_vy);
				fy_vec[i] = CppAD::Value(forces[1]);
			}
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << CppAD::Value(tire_vx);
			std::string label = stream.str();
		  
			float grey = (0.8*j) / num;
			plt::plot(vy_vec, fy_vec, {{"color", std::to_string(grey)}, {"label", label}});
		}

		plt::legend();
		plt::xlabel("Lateral Velocity (m/s)");
		plt::ylabel("Lateral Force (N)");
		plt::show();
	}

	TEST_F(BekkerFixture, vz_fz_plot)
	{
		BekkerTireModel tire_model;
	  
		Eigen::Matrix<ADF,8,1> features;
		Eigen::Matrix<ADF,4,1> forces;
	  
		features[0] = 0;
		features[1] = 0;
		features[2] = 0;
	  
		features[3] = 29.758547;
		features[4] = 2083.0;
		features[5] = 1.197933;
		features[6] = 0.102483;
		features[7] = 0.652405;
    
		int len = 1000;
		std::vector<float> vz_vec(len);
		std::vector<float> fz_vec(len);

		ADF tire_tangent_vel = 0.0;
		ADF tire_vx = 0.0;
		for(int i = 0; i < len; i++)
		{
			ADF zr = 0.01 * ADF((2.0*i/(float)len) - 0.5);
			
			std::cout << "zr: " << CppAD::Value(zr) << "\n";
			
			features[0] = zr;
			features[1] = 1;
			features[2] = 0;
			  
			forces = tire_model.get_forces(features);
			  
			vz_vec[i] = CppAD::Value(zr);
			fz_vec[i] = CppAD::Value(forces[2]);
		}
      
		plt::plot(vz_vec, fz_vec, {{"color", "k"}});
		plt::xlabel("Tire Contact Height Error (m)");
		plt::ylabel("Normal Force (N)");
		plt::show();
	}

	TEST_F(BekkerFixture, settling)
	{
		int num_steps = 100;
		std::vector<double> time(num_steps);
		std::vector<double> elev(num_steps);
		std::vector<double> x_vec(num_steps);
		std::vector<double> y_vec(num_steps);

		m_system_adf->setParams(m_params);

		VectorAD xk(m_system_adf->getStateDim() + m_system_adf->getControlDim());
		VectorAD xk1(m_system_adf->getStateDim() + m_system_adf->getControlDim());
    
		xk = m_x0;
    
		for(int i = 0; i < num_steps; i++)
		{
			xk[m_system_adf->getStateDim()] = 0.0; //vl
			xk[m_system_adf->getStateDim()+1] = 0.0; //vr
			m_system_adf->integrate(xk, xk1);
      
			time[i] = i * 0.01;
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

	TEST_F(BekkerFixture, no_lateral_drift)
	{
		int num_steps = 1000;
		std::vector<double> time(num_steps);
		std::vector<double> elev(num_steps);
		std::vector<double> x_vec(num_steps);
		std::vector<double> y_vec(num_steps);
		
		m_system_adf->setParams(m_params);
		
		VectorAD xk(m_system_adf->getStateDim() + m_system_adf->getControlDim());
		VectorAD xk1(m_system_adf->getStateDim() + m_system_adf->getControlDim());
		
		xk = m_x0;
		xk[15] = .1; //add an initial lateral movement
		
		for(int i = 0; i < num_steps; i++)
		{
			xk[m_system_adf->getStateDim()] = 0.0; //vl
			xk[m_system_adf->getStateDim()+1] = 0.0; //vr
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
	
	
	// Moving straight at 4rad/s, tire radius is .098, total time is 10s
	// Expected distance is .392m.
	TEST_F(BekkerFixture, straight)
	{
		int num_steps = 1000;
		std::vector<double> time(num_steps);
		std::vector<double> elev(num_steps);
		std::vector<double> x_vec(num_steps);
		std::vector<double> y_vec(num_steps);
    
		m_system_adf->setParams(m_params);
		
		VectorAD xk(m_system_adf->getStateDim() + m_system_adf->getControlDim());
		VectorAD xk1(m_system_adf->getStateDim() + m_system_adf->getControlDim());
		
		xk = m_x0;
    
		for(int i = 0; i < num_steps; i++)
		{
			xk[m_system_adf->getStateDim()] = 4; //vl
			xk[m_system_adf->getStateDim()+1] = 4; //vr
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
		EXPECT_NEAR(x_vec.back(), 10*.098*4, 1e-1);
	}

	TEST_F(BekkerFixture, circle)
	{
		int num_steps = 1000;
		std::vector<double> time(num_steps);
		std::vector<double> elev(num_steps);
		std::vector<double> x_vec(num_steps);
		std::vector<double> y_vec(num_steps);
    
		m_system_adf->setParams(m_params);
		
		VectorAD xk(m_system_adf->getStateDim() + m_system_adf->getControlDim());
		VectorAD xk1(m_system_adf->getStateDim() + m_system_adf->getControlDim());
		
		xk = m_x0;

		double max_x = CppAD::Value(xk[4]);
		double min_x = CppAD::Value(xk[4]);
		double max_y = CppAD::Value(xk[5]);
		double min_y = CppAD::Value(xk[5]);
    
		for(int i = 0; i < num_steps; i++)
		{
			xk[m_system_adf->getStateDim()+0] = 2; //vl
			xk[m_system_adf->getStateDim()+1] = -2; //vr
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

}

