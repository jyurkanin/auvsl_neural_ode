#include <vector>
#include <map>
#include <string>
#include <matplotlibcpp.h>

#include "HybridDynamics.h"
#include "VehicleSystem.h"
#include "TestTerrainMaps.h"
#include "utils.h"

#include "gtest/gtest.h"

namespace plt = matplotlibcpp;

namespace
{
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorF;
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixF;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, 1> VectorAD;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> MatrixAD;
	
	class TerrainFixture : public ::testing::Test
	{
	public:
		VectorAD m_x0;
		VectorAD m_params;

		std::shared_ptr<const BumpyTerrainMap<ADF>> m_map;
		std::shared_ptr<VehicleSystem<ADF>> m_system_adf;
		
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
		
		TerrainFixture()
		{
			srand(time(NULL)); // randomize seed
			
			m_map = std::make_shared<const BumpyTerrainMap<ADF>>();
			m_system_adf = std::make_shared<VehicleSystem<ADF>>(m_map);
			
			m_params = VectorAD::Zero(m_system_adf->getNumParams());
			m_x0 = VectorAD::Zero(m_system_adf->getStateDim() + m_system_adf->getControlDim());
			
			if(loadVec(m_params, "/home/justin/tire.net"))
			{
				m_system_adf->getDefaultParams(m_params);
			}
			
			m_system_adf->setParams(m_params);
			
			m_system_adf->getDefaultInitialState(m_x0);
		}
		~TerrainFixture(){}

		void checkSafety(double vl,
						 double vr,
						 double &max_angle,
						 int &len,
						 std::vector<double> &x_vec,
						 std::vector<double> &y_vec,
						 std::vector<double> &z_vec,
						 std::vector<double> &time_vec)
		{
			VectorAD xk(m_system_adf->getStateDim() + m_system_adf->getControlDim());
			VectorAD xk1(m_system_adf->getStateDim() + m_system_adf->getControlDim());
		
			xk = m_x0;

			max_angle = 0.0;
			for(int i = 0; i < time_vec.size(); i++)
			{
				xk[HybridDynamics::STATE_DIM] = vl;
				xk[HybridDynamics::STATE_DIM+1] = vr;
				m_system_adf->integrate(xk, xk1);
      
				time_vec[i] = i * 0.01;
				x_vec[i] = CppAD::Value(xk1[4]);
				y_vec[i] = CppAD::Value(xk1[5]);
				z_vec[i] = CppAD::Value(xk1[6]);
				xk = xk1;
				
				ADF qx = xk[0];
				ADF qy = xk[1];
				ADF qz = xk[2];
				ADF qw = xk[3];				
				Eigen::Matrix<ADF,3,3> rot = toMatrixRotation(qx, qy, qz, qw);
				Eigen::Matrix<ADF,3,1> z_vec;
				Eigen::Matrix<ADF,3,1> rot_z_vec;
				z_vec[0] = 0;
				z_vec[1] = 0;
				z_vec[2] = 1;
				
				rot_z_vec = rot*z_vec;

				double angle = CppAD::Value(CppAD::acos(rot_z_vec[2]));
				
				max_angle = std::max(angle, max_angle);
				len = i+1;
				
				if(max_angle > 0.7853)
				{
					return;
				}
			}
		
	}

	};

	TEST_F(TerrainFixture, settling)
	{
		int num_steps = 100;
		std::vector<double> time(num_steps);
		std::vector<double> elev(num_steps);
		std::vector<double> x_vec(num_steps);
		std::vector<double> y_vec(num_steps);
    
		VectorAD xk(m_system_adf->getStateDim() + m_system_adf->getControlDim());
		VectorAD xk1(m_system_adf->getStateDim() + m_system_adf->getControlDim());
    
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
	
	
	
	TEST_F(TerrainFixture, straight)
	{
		int num_steps = 1000;
		std::vector<double> time(num_steps);
		std::vector<double> elev(num_steps);
		std::vector<double> x_vec(num_steps);
		std::vector<double> y_vec(num_steps);
    
		VectorAD xk(m_system_adf->getStateDim() + m_system_adf->getControlDim());
		VectorAD xk1(m_system_adf->getStateDim() + m_system_adf->getControlDim());
		
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
	
	TEST_F(TerrainFixture, safety_colormap)
	{
		const int max_i = 10;
 		const int max_j = 10;
		int num_steps = 1000;
		double vel_scale = 10.0;
		
		std::vector<double> time_vec(num_steps);
		std::vector<double> x_vec(num_steps);
		std::vector<double> y_vec(num_steps);
		std::vector<double> z_vec(num_steps);
		Eigen::Matrix<double, max_i, max_j>  max_angle_mat;
		
		plt::figure(1);
		plt::title("X-Y plot");
		plt::xlabel("[m]");
		plt::ylabel("[m]");

		plt::figure(2);
		plt::title("Elevation vs Time");
		plt::xlabel("[s]");
		plt::ylabel("[m]");
		
		for(int i = 0; i < max_i; i++)
		{
			for(int j = 0; j < max_j; j++)
			{
				std::cout << "Pixel: " << i << ", " << j << "\n";
				
				double vl = vel_scale*i/max_i;
				double vr = vel_scale*j/max_j;
				double max_angle;
				int len;
				
				checkSafety(vl, vr, max_angle, len,
							x_vec, y_vec, z_vec, time_vec);
				
				max_angle_mat(max_i-1-i,j) = max_angle;
				
				plt::figure(1);
				std::string format;
				if(len < time_vec.size())
				{
					format = "red";
				}
				else
				{
					format = "black";
				}
				plt::plot(std::vector<double>(x_vec.begin(), x_vec.begin()+len),
						  std::vector<double>(y_vec.begin(), y_vec.begin()+len),
						  format);
				
				plt::figure(2);
				plt::plot(std::vector<double>(time_vec.begin(), time_vec.begin()+len),
						  std::vector<double>(z_vec.begin(), z_vec.begin()+len),
						  format);
			}
		}
		
		
		
		plt::figure(3);
		plt::title("Safety of Different Vehicle Trajectories");
		
		const std::map<std::string, std::string> kwds =
			{{"interpolation", "none"},
			 {"cmap","hot"}
			};
		plt::imshow(max_angle_mat, kwds);
		plt::colorbar();
		
		const std::vector<double> xticks_safety = {0,1,2,3,4,5,6,7,8,9};
		const std::vector<double> yticks_safety = {9,8,7,6,5,4,3,2,1,0};
		const std::vector<std::string> labels_safety = {"0","1", "2", "3", "4", "5", "6", "7", "8", "9"};
		plt::xticks(xticks_safety, labels_safety);
		plt::yticks(yticks_safety, labels_safety);
		plt::xlabel("vr");
		plt::ylabel("vl");
		
		
		
		plt::figure(4);
		plt::title("Simulated Map Elevation");

		const int num_rows = 100;
		const int num_cols = 100;
		const double scale = 10.0;
		const std::vector<double> xticks = {0,9,19,29,39,49,59,69,79,89,99};
		const std::vector<double> yticks = {99,89,79,69,59,49,39,29,19,9,0};
		const std::vector<std::string> labels = {"-5","-4","-3","-2","-1","0","1","2","3","4","5"};
		plt::xticks(xticks, labels);
		plt::yticks(yticks, labels);
		plt::xlabel("x position [m]");
		plt::ylabel("y position [m]");
		
		Eigen::Matrix<double, num_rows, num_cols> elev_mat;
		for(int i = 0; i < num_rows; i++)
		{
			for(int j = 0; j < num_cols; j++)
			{
				ADF x = scale*(((double)j/num_cols) - 0.5);
				ADF y = scale*(((double)i/num_rows) - 0.5);
				elev_mat(num_rows-1-i,j) = CppAD::Value(m_map->getAltitude(x, y));
			}
		}
		
		plt::imshow(elev_mat, kwds);

		plt::figure(5);
		std::vector<double> x_map_vec(1000);
		std::vector<double> z_map_vec(1000);
		for(int i = 0; i < 1000; i++)
		{
			x_map_vec[i] = 10.0*i / 1000;
			z_map_vec[i] = CppAD::Value(m_map->getAltitude(x_map_vec[i], 0.0));
		}
		plt::plot(x_map_vec, z_map_vec);
		
		plt::show();
	}
}
