#include "BekkerSystem.h"
#include "VehicleSystem.h"
#include "TestTerrainMaps.h"
#include "Trainer.h"

#include "types/GroundTruthDataRow.h"
#include "types/Scalars.h"

#include <iostream>
#include <string>
#include <matplotlibcpp.h>
#include <Eigen/Dense>

namespace plt = matplotlibcpp;
typedef Eigen::Matrix<ADF, Eigen::Dynamic, 1> VectorAD;

int plot_stuff()
{
	int num_threads = 1;
	
	auto map = std::make_shared<const FlatTerrainMap<ADF>>();
	auto bekker_factory = std::make_shared<BekkerSystemFactory<ADF>>(map);
	auto neural_factory = std::make_shared<VehicleSystemFactory<ADF>>(map);

	Trainer train_bekker(bekker_factory, num_threads);
	Trainer train_neural(neural_factory, num_threads);
	train_neural.load();
	
	const std::string ex1_name = "/home/justin/code/auvsl_dynamics_bptt/scripts/CV3_data34.csv";
	const std::string ex2_name = "/home/justin/code/auvsl_dynamics_bptt/scripts/CV3_data55.csv";
	const std::string ex3_name = "/home/justin/code/auvsl_dynamics_bptt/scripts/LD3_data01.csv";
	
	std::string names[3] = {ex1_name, ex2_name, ex3_name};
	for(int i = 0; i < 3; i++)
	{		
		std::string name = names[i];
		
		std::vector<VectorAD> nn_traj;
		std::vector<VectorAD> bk_traj;
		std::vector<GroundTruthDataRow> gt_traj;
		std::vector<GroundTruthDataRow> gt_traj_ignore;
		
		train_bekker.evaluate_file(name,
								   1,
								   bk_traj,
								   gt_traj);
		train_neural.evaluate_file(name,
								   1,
								   nn_traj,
								   gt_traj_ignore);
		
		std::vector<double> nn_x_pos(nn_traj.size());
		std::vector<double> nn_y_pos(nn_traj.size());
		std::vector<double> bk_x_pos(bk_traj.size());
		std::vector<double> bk_y_pos(bk_traj.size());
		std::vector<double> gt_x_pos(gt_traj.size());
		std::vector<double> gt_y_pos(gt_traj.size());
		
		std::transform(nn_traj.begin(), nn_traj.end(), nn_x_pos.begin(), [](const VectorAD &vec){return CppAD::Value(vec[4]);});
		std::transform(nn_traj.begin(), nn_traj.end(), nn_y_pos.begin(), [](const VectorAD &vec){return CppAD::Value(vec[5]);});
		std::transform(bk_traj.begin(), bk_traj.end(), bk_x_pos.begin(), [](const VectorAD &vec){return CppAD::Value(vec[4]);});
		std::transform(bk_traj.begin(), bk_traj.end(), bk_y_pos.begin(), [](const VectorAD &vec){return CppAD::Value(vec[5]);});
		std::transform(gt_traj.begin(), gt_traj.end(), gt_x_pos.begin(), [](const GroundTruthDataRow &row){return row.x;});
		std::transform(gt_traj.begin(), gt_traj.end(), gt_y_pos.begin(), [](const GroundTruthDataRow &row){return row.y;});
		
		plt::figure(i+1);
		//plt::title(name);
		plt::xlabel("x (m)");
		plt::ylabel("y (m)");
		plt::plot(nn_x_pos, nn_y_pos, {{"color", "0.0"}, {"label", "Neural"}});
		plt::plot(bk_x_pos, bk_y_pos, {{"color", "0.4"}, {"label", "Bekker"}});
		plt::plot(gt_x_pos, gt_y_pos, {{"color", "0.8"}, {"label", "GroundTruth"}});
		plt::legend();
	}
	
	plt::show();
}



void time_stuff()
{
	int num_threads = 1;
	
	auto map = std::make_shared<const FlatTerrainMap<ADF>>();
	auto bekker_factory = std::make_shared<BekkerSystemFactory<ADF>>(map);
	auto neural_factory = std::make_shared<VehicleSystemFactory<ADF>>(map);
	
	Trainer train_bekker(bekker_factory, num_threads);
	Trainer train_neural(neural_factory, num_threads);
	train_neural.load();
	
	const std::string ex3_name = "/home/justin/code/auvsl_dynamics_bptt/scripts/LD3_data01.csv";
	
	std::vector<VectorAD> nn_traj;
	std::vector<VectorAD> bk_traj;
	std::vector<GroundTruthDataRow> gt_traj;
	std::vector<GroundTruthDataRow> gt_traj_ignore;

	const int num_steps = 100;
	const float num_secs = num_steps*6;


	auto start = std::chrono::system_clock::now();
	train_bekker.evaluate_file(ex3_name,
							   num_steps,
							   bk_traj,
							   gt_traj);	
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> bekker_runtime = end - start;
	

	start = std::chrono::system_clock::now();
	train_neural.evaluate_file(ex3_name,
							   num_steps,
							   nn_traj,
							   gt_traj_ignore);
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> neural_runtime = end - start;
	
	
	std::cout << "Bekker 600s of Simulation in: " << bekker_runtime.count() << "s\n";
	std::cout << "Neural 600s of Simulation in: " << neural_runtime.count() << "s\n";
}



int main()
{
	// plot_stuff();
	time_stuff();
}
