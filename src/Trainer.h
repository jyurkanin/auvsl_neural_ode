#pragma once

#include <cpp_bptt.h>

#include "VehicleSystem.h"
#include <string>

struct DataRow
{
  double time;
  float vl;
  float vr;
  float x;
  float y;
  float yaw;
  float wz;
  float vx;
  float vy;
};



class Trainer
{
public:
  Trainer();
  ~Trainer();

  void updateParams(const VectorF &grad);
  void evaluateTrajectory(const std::vector<DataRow> &traj, std::vector<VectorAD> &x_list, float &loss);
  void trainTrajectory(const std::vector<DataRow> &traj, std::vector<VectorAD> &x_list, VectorF &gradient, float& loss);
  void plotTrajectory(const std::vector<DataRow> &traj, const std::vector<VectorAD> &x_list);
  void initializeState(const DataRow &gt_state, VectorAD &xk_robot);
  void loadDataFile(std::string string);
  void train();
  void evaluate();
  void computeEqState();
  
private:
  const float m_gt_sample_period = 10.0f;
  
  ADF m_z_stable;
  VectorAD m_quat_stable;
  
  int m_cnt;
  std::vector<DataRow> m_data;
  VectorAD m_params;
  std::shared_ptr<VehicleSystem<ADF>> m_system_adf;
};
