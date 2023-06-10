#pragma once

#include <cpp_bptt.h>

#include "VehicleSystem.h"
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



class Trainer
{
public:
  friend class Worker;
  
  Trainer(int num_threads = 2);
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

  void trainThreads();
  void assignWork(const std::vector<DataRow> &traj);
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
    std::vector<DataRow> m_traj;
    
    // results:
    VectorF m_grad;
    double m_loss;

  private:
    Trainer* m_trainer;
  };

  std::vector<Worker> m_workers;
private:
  const int m_train_steps = 100;
  const int m_eval_steps = 600;
  const int m_inc_train_steps = 100;
  const int m_inc_eval_steps = 600;
  const double m_l1_weight = 1e-3f;
  
  const int m_num_threads;
  
  ADF m_z_stable;
  VectorAD m_quat_stable;

  std::string m_param_file;
  int m_cnt;
  std::vector<DataRow> m_data;
  VectorAD m_params;
  VectorAD m_squared_grad;
  VectorF m_batch_grad;
  double m_batch_loss;
  std::shared_ptr<VehicleSystem<ADF>> m_system_adf;
};
