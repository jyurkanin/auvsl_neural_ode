#pragma once

#include <cpp_bptt.h>

#include "VehicleSystem.h"
#include <string>

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



class Trainer
{
public:
  Trainer();
  ~Trainer();

  void updateParams(const VectorF &grad);
  void evaluateTrajectory(const std::vector<DataRow> &traj, std::vector<VectorAD> &x_list, double &loss);
  void trainTrajectory(const std::vector<DataRow> &traj, std::vector<VectorAD> &x_list, VectorF &gradient, double& loss);
  void plotTrajectory(const std::vector<DataRow> &traj, const std::vector<VectorAD> &x_list);
  void initializeState(const DataRow &gt_state, VectorAD &xk_robot);
  void loadDataFile(std::string string);
  void train();
  void evaluate_cv3();
  void evaluate_ld3();
  void computeEqState();
  bool saveVec(const VectorAD &params, const std::string &file_name);
  bool loadVec(VectorAD &params, const std::string &file_name);
  void save();
  void load();
  
private:
  const double m_gt_sample_period = 10.0;
  
  ADF m_z_stable;
  VectorAD m_quat_stable;

  const int m_train_steps = 2;
  const int m_eval_steps = 60;
  
  std::string m_param_file;
  int m_cnt;
  std::vector<DataRow> m_data;
  VectorAD m_params;
  VectorAD m_squared_grad;
  VectorF m_batch_grad;
  std::shared_ptr<VehicleSystem<ADF>> m_system_adf;
};
