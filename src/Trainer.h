#pragma once

#include "types/System.h"
#include "types/TerrainMap.h"
#include "types/SystemFactory.h"
#include "types/GroundTruthDataRow.h"

#include <thread>
#include <string>
#include <atomic>
#include <vector>

class Trainer
{
public:
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorF;
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixF;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, 1> VectorAD;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> MatrixAD;
	
	friend class Worker;
  
	Trainer(std::shared_ptr<SystemFactory<ADF>> factory,
			int num_threads = 2);
	~Trainer();

	void updateParams(const VectorF &grad);
	void evaluateTrajectory(const std::vector<GroundTruthDataRow> &traj, std::vector<VectorAD> &x_list, double &loss);
	void trainTrajectory(const std::vector<GroundTruthDataRow> &traj, std::vector<VectorAD> &x_list, VectorF &gradient, double& loss);
	void plotTrajectory(const std::vector<GroundTruthDataRow> &traj, const std::vector<VectorAD> &x_list);
	void loadDataFile(std::string string);
	void train();
	void evaluate_train3();
	void evaluate_cv3();
	void evaluate_ld3();
	void evaluate_validation_dataset();
	void computeEqState();
	bool saveVec(const VectorAD &params, const std::string &file_name);
	bool loadVec(VectorAD &params, const std::string &file_name);
	void save();
	void load();

	void trainThreads();
	void assignWork(const std::vector<GroundTruthDataRow> &traj);
	void finishWork();
	bool combineResults(VectorF &batch_grad,
						const VectorF &sample_grad,
						double &batch_loss,
						const double &sample_loss);
  
	class Worker
	{
	public:
		Worker();
		Worker(const Worker& other);
		Worker(Trainer* trainer);
		~Worker();

		const Worker& operator=(const Worker& worker);
		void work();
    
		// Worker State:
		std::atomic<bool> m_keep_alive{false};

		std::atomic<bool> m_idle{true};
		std::atomic<bool> m_ready{true};
		std::atomic<bool> m_waiting{true};
		std::thread m_thread;
		int m_id;
    
		// Problem Definition:
		std::vector<GroundTruthDataRow> m_traj;
    
		// results:
		VectorF m_grad;
		double m_loss;

	private:
		Trainer* m_trainer;
	};

	std::vector<Worker> m_workers;
	
	//private: // lol!
	const int m_train_steps = 200;
	const int m_eval_steps = 600;
	const int m_inc_train_steps = 200;
	const int m_inc_eval_steps = 600;
	const double m_l1_weight = 0.0;
	const double m_lr = 1e-3;
	
	double m_best_CV3;
	
	ADF m_z_stable;
	VectorAD m_quat_stable;

	std::string m_param_file;
	int m_cnt;
	std::vector<GroundTruthDataRow> m_data;
	VectorAD m_params;
	VectorAD m_squared_grad;
	VectorF m_batch_grad;
	double m_batch_loss;
	
	std::shared_ptr<SystemFactory<ADF>> m_factory_adf;
	std::shared_ptr<System<ADF>> m_system_adf;
	
	const int m_num_threads;
};
