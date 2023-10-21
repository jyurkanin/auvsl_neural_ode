#include <vector>
#include "gtest/gtest.h"

#include "BekkerSystem.h"
#include "VehicleSystem.h"
#include "Trainer.h"
#include "TestTerrainMaps.h"

#include <matplotlibcpp.h>
#include <memory>
#include <iostream>

namespace plt = matplotlibcpp;

namespace {
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorF;
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixF;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, 1> VectorAD;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> MatrixAD;
	
	class TrainerFixture : public ::testing::Test
	{
	public:
		TrainerFixture()
		{
			srand(time(NULL)); //randomize the seed
			
			auto map = std::make_shared<const FlatTerrainMap<ADF>>();
			auto factory = std::make_shared<VehicleSystemFactory<ADF>>(map);
			m_trainer = std::make_shared<Trainer>(factory);
			m_trainer->load();

			auto bk_factory = std::make_shared<BekkerSystemFactory<ADF>>(map);
			m_trainer_bekker = std::make_shared<Trainer>(bk_factory);
			m_trainer_bekker->load();
		}

		// Lol this is really bad.
		void evaluate_cv3_traj(int traj_num)
		{
			std::vector<VectorAD> x_list(m_trainer->m_eval_steps);
			std::vector<VectorAD> x_list_bekker(m_trainer->m_eval_steps);
			
			char fn_array[100];
  
			double loss_avg = 0;
			double loss = 0;
			int cnt = 0;
  
			memset(fn_array, 0, 100);
			sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/CV3_data%02d.csv", traj_num);
    
			std::string fn(fn_array);
			m_trainer->loadDataFile(fn);
			
			bool first_time = true;
			
			plt::figure(traj_num);
			plt::title("Model Trajectory vs Ground Truth");
			for(int j = 0; j < (m_trainer->m_data.size() - m_trainer->m_eval_steps); j += m_trainer->m_inc_eval_steps)
			{
				std::vector<double> x_vec_model;
				std::vector<double> y_vec_model;
				std::vector<double> x_vec_bekker;
				std::vector<double> y_vec_bekker;
				std::vector<double> x_vec_gt;
				std::vector<double> y_vec_gt;

				
				std::vector<GroundTruthDataRow> traj(m_trainer->m_data.begin()+j, m_trainer->m_data.begin()+j+m_trainer->m_eval_steps);
				m_trainer->evaluateTrajectory(traj, x_list, loss);
				m_trainer_bekker->evaluateTrajectory(traj, x_list_bekker, loss);
				
				std::transform(traj.begin(), traj.end(), std::back_inserter(x_vec_gt), [](const GroundTruthDataRow &row){return row.x;});
				std::transform(traj.begin(), traj.end(), std::back_inserter(y_vec_gt), [](const GroundTruthDataRow &row){return row.y;});
				
				std::transform(x_list.begin(), x_list.end(), std::back_inserter(x_vec_model), [](const VectorAD &state){return CppAD::Value(state[4]);});
				std::transform(x_list.begin(), x_list.end(), std::back_inserter(y_vec_model), [](const VectorAD &state){return CppAD::Value(state[5]);});

				std::transform(x_list_bekker.begin(), x_list_bekker.end(), std::back_inserter(x_vec_bekker), [](const VectorAD &state){return CppAD::Value(state[4]);});
				std::transform(x_list_bekker.begin(), x_list_bekker.end(), std::back_inserter(y_vec_bekker), [](const VectorAD &state){return CppAD::Value(state[5]);});

				if(first_time)
				{
					plt::plot(x_vec_model, y_vec_model, {{"color", "b"}, {"label","Neural Model"}});
					plt::plot(x_vec_bekker, y_vec_bekker, {{"color", "g"}, {"label","Bekker Model"}});
					plt::plot(x_vec_gt, y_vec_gt, {{"color", "r"}, {"label","Ground Truth"}});
					first_time = false;
				}
				else
				{
					plt::plot(x_vec_model, y_vec_model);
					plt::plot(x_vec_bekker, y_vec_bekker);
					plt::plot(x_vec_gt, y_vec_gt);
				}
				
				std::cout << "CV3 " << traj_num << " loss: " << loss << "\n;";
				
				loss_avg += loss;
				cnt++;
			}

			plt::legend();
		}
		
		std::shared_ptr<Trainer> m_trainer;
		std::shared_ptr<Trainer> m_trainer_bekker;
	};
	
	
	
	TEST_F(TrainerFixture, load_save)
	{
		std::string file_name = "temp.txt";
		VectorAD init_params(100);
		VectorAD temp_params(100);
		for(int i = 0; i < init_params.size(); i++)
		{
			init_params[i] = i;
		}
		
		m_trainer->saveVec(init_params, file_name);
		m_trainer->loadVec(temp_params, file_name);

		for(int i = 0; i < temp_params.size(); i++)
		{
			EXPECT_EQ(init_params[i], temp_params[i]);
		}
	}
	
	TEST_F(TrainerFixture, plot_cv3_special)
	{
		evaluate_cv3_traj(1);
		evaluate_cv3_traj(55);
		evaluate_cv3_traj(26);

		plt::show();
	}

}
