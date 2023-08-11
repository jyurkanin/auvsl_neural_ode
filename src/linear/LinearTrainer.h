#pragma once

#include <cpp_bptt.h>

#include "linear/LinearSystem.h"
#include <string>
#include <atomic>
#include <vector>

struct DataRow
{
  double time;
  double vl;
  double vr;
  double x;
  double y;
  double yaw;
  double wz;
  double vx;
  double vy;
};



class LinearTrainer
{
public:
	LinearTrainer();
	~LinearTrainer();

	void updateParams(const VectorF &grad);
	void evaluateTrajectory(const std::vector<DataRow> &traj, std::vector<VectorAD> &x_list, double &loss);
	void plotTrajectory(const std::vector<DataRow> &traj, const std::vector<VectorAD> &x_list);
	void initializeState(const DataRow &gt_state, VectorAD &xk_robot);
	void loadDataFile(std::string string);
	void train();
	void evaluate_train3();
	void evaluate_cv3();
	void evaluate_ld3();
	bool saveVec(const VectorAD &params, const std::string &file_name);
	bool loadVec(VectorAD &params, const std::string &file_name);
	void save();
	void load();
	
private:
	const int m_eval_steps = 600;
	const int m_inc_eval_steps = 600;
	
	std::string m_param_file;
	int m_cnt;
	std::vector<DataRow> m_data;
	VectorAD m_params;
	VectorAD m_squared_grad;
	VectorF m_batch_grad;
	double m_batch_loss;
	std::shared_ptr<LinearSystem<ADF>> m_system_adf;
};
