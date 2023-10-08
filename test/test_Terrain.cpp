#include <vector>
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
			
			auto map = std::make_shared<const BumpyTerrainMap<ADF>>();
			m_system_adf = std::make_shared<VehicleSystem<ADF>>(map);
			
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
}
