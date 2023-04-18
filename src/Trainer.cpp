#include "Trainer.h"

#include <matplotlibcpp.h>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <cmath>
#include <stdlib.h>

namespace plt = matplotlibcpp;


Trainer::Trainer()
{
  m_system_adf = std::make_shared<VehicleSystem<ADF>>();
 
  m_params = VectorAD::Zero(m_system_adf->getNumParams());
  m_batch_grad = VectorF::Zero(m_system_adf->getNumParams());
  m_squared_grad = VectorAD::Ones(m_system_adf->getNumParams());
  m_system_adf->getDefaultParams(m_params);

  computeEqState();
  m_cnt = 0;
  m_param_file = "/home/justin/tire.net";
}

Trainer::~Trainer()
{

}

void Trainer::computeEqState()
{
  m_system_adf->m_hybrid_dynamics.initState(); //set start pos to 0,0,.16 and orientation to 0,0,0,1
  m_system_adf->m_hybrid_dynamics.settle();     //allow the 3d vehicle to come to rest and reach steady state, equillibrium sinkage for tires.

  m_quat_stable = VectorAD::Zero(4);
  m_quat_stable[0] = m_system_adf->m_hybrid_dynamics.state_[0];
  m_quat_stable[1] = m_system_adf->m_hybrid_dynamics.state_[1];
  m_quat_stable[2] = m_system_adf->m_hybrid_dynamics.state_[2];
  m_quat_stable[3] = m_system_adf->m_hybrid_dynamics.state_[3];
  
  m_z_stable = m_system_adf->m_hybrid_dynamics.state_[6];

}

// ,time,vel_left,vel_right,x,y,yaw,wx,wy,wz
void Trainer::loadDataFile(std::string fn)
{
  std::cout << "Opening " << fn << "\n";
  std::flush(std::cout);
  
  std::ifstream data_file(fn);

  if(!data_file.is_open())
  {
    std::cout << "File was not open\n";
  }
  
  std::string line;
  std::getline(data_file, line); //ignore column heading
  
  char comma;
  int idx;
  double wx, wy;
  DataRow row;

  m_data.clear();
  while(data_file.peek() != EOF)
  {
    data_file >> idx >> comma;
    data_file >> row.time >> comma;
    data_file >> row.vl >> comma;
    data_file >> row.vr >> comma;
    data_file >> row.x >> comma;
    data_file >> row.y >> comma;
    data_file >> row.yaw >> comma;
    data_file >> wx >> comma; //ignore this
    data_file >> wy >> comma; //ignore this
    data_file >> row.wz >> comma;
    data_file >> row.vx >> comma;
    data_file >> row.vy; // >> comma;
    
    m_data.push_back(row);
  }
  
}

void Trainer::train()
{
  m_system_adf->setNumSteps(m_train_steps);
  
  VectorF traj_grad(m_system_adf->getNumParams());
  int traj_len = m_system_adf->getNumSteps();
  double loss;
  double avg_loss = 0;
  std::vector<VectorAD> x_list(m_system_adf->getNumSteps());
  char fn_array[100];
  
  for(int i = 1; i <= 17; i++)
  {
    memset(fn_array, 0, 100);
    sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/Train3_data%02d.csv", i);
    
    std::string fn(fn_array);
    loadDataFile(fn);
    
    for(int j = 0; j < (m_data.size() - traj_len); j += traj_len)
    {
      std::vector<DataRow> traj(m_data.begin() + j, m_data.begin() + j + traj_len);
      trainTrajectory(traj, x_list, traj_grad, loss);
      
      bool has_explosion = false;
      for(int i = 0; i < m_params.size(); i++)
      {
	if(fabs(traj_grad[i]) > 100.0)
	{
	  has_explosion = true;
	  std::cout << "Explosion " << i << ":" << traj_grad[i] << "\n";
	  break;
	}
      }

      if(!has_explosion)
      {
	m_batch_grad += traj_grad;
	avg_loss += loss;
	//plotTrajectory(traj, x_list);      
	std::cout << "Loss: " << loss << "\tdParams: " << traj_grad[0] << "\n";
	std::flush(std::cout);
	m_cnt++;
      }
      
      if(m_cnt == 10)
      {
	std::cout << "Avg Loss: " << avg_loss / m_cnt << ", Batch Grad[0]: " << m_batch_grad[0] << "\n";
	std::flush(std::cout);
	updateParams(m_batch_grad / m_cnt);
	m_cnt = 0;
	avg_loss = 0;
      }
    }

    save();
  }
  
}

void Trainer::evaluate_cv3()
{
  std::vector<VectorAD> x_list(m_system_adf->getNumSteps());
  int traj_len = m_system_adf->getNumSteps();
  char fn_array[100];
  
  double loss_avg = 0;
  double loss = 0;
  int cnt = 0;
  
  m_system_adf->setNumSteps(m_eval_steps);
  
  for(int i = 1; i <= 144; i++)
  {
    memset(fn_array, 0, 100);
    sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/CV3_data%02d.csv", i);
    
    std::string fn(fn_array);
    loadDataFile(fn);
    
    for(int j = 0; j < (m_data.size() - traj_len); j += traj_len)
    {
      std::vector<DataRow> traj(m_data.begin()+j, m_data.begin()+j+traj_len);
      evaluateTrajectory(traj, x_list, loss);
      //plotTrajectory(traj, x_list);
      
      loss_avg += loss;
      cnt++;
    }
  }
  
  std::cout << "CV3 avg loss: " << loss_avg/cnt << "\n";
}

void Trainer::evaluate_ld3()
{
  std::vector<VectorAD> x_list(m_system_adf->getNumSteps());
  int traj_len = m_system_adf->getNumSteps();
  char fn_array[100];
  
  double loss_avg = 0;
  double loss = 0;
  int cnt = 0;
  
  m_system_adf->setNumSteps(m_eval_steps);
  
  memset(fn_array, 0, 100);
  sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/LD3_data%02d.csv", 1);
  std::string fn(fn_array);

  loadDataFile(fn);
  for(int j = 0; j < (m_data.size() - traj_len); j += traj_len)
  {
    std::vector<DataRow> traj(m_data.begin()+j, m_data.begin()+j+traj_len);
    evaluateTrajectory(traj, x_list, loss);
    //plotTrajectory(traj, x_list);
    
    loss_avg += loss;
    cnt++;
  }
  
  std::cout << "LD3 avg loss: " << loss_avg/cnt << "\n";
}

void Trainer::updateParams(const VectorF &grad)
{
  ADF norm = 0;
  float grad_idx;
  
  for(int i = 0; i < m_params.size(); i++)
  {
    grad_idx = grad[i];
    if(grad[i] < -1)
    {
      std::cout << "Clipped [" << i << "] " << grad[i] << "\n";
      grad_idx = -1;
    }
    else if(grad[i] > 1)
    {
      std::cout << "Clipped [" << i << "] " << grad[i] << "\n";
      grad_idx = 1;
    }
    
    m_squared_grad[i] = 0.9*m_squared_grad[i] + 0.1*ADF(grad_idx*grad_idx);
    m_params[i] -= (m_system_adf->getLearningRate()/(CppAD::sqrt(m_squared_grad[i]) + 1e-6))*ADF(grad_idx);
    norm += CppAD::abs(m_params[i]);
  }
  
  std::cout << "Param norm: " << CppAD::Value(norm) << " Param[0]: " << CppAD::Value(m_params[0]) << "\n";
  for(int i = 0; i < m_params.size(); i++)
  {
    
    if(CppAD::abs(m_params[i]) > 100.0)
    {
      std::cout << "Param[" << i <<"] Exploded: " << CppAD::Value(m_params[i]) << "\n";
      break;
    }
  }

  m_batch_grad = VectorF::Zero(m_system_adf->getNumParams());
}

void Trainer::plotTrajectory(const std::vector<DataRow> &traj, const std::vector<VectorAD> &x_list)
{
  assert(traj.size() == x_list.size());
  std::vector<double> model_x(x_list.size());
  std::vector<double> model_y(x_list.size());
  std::vector<double> model_z(x_list.size());
  std::vector<double> model_yaw(x_list.size());
  
  std::vector<double> gt_x(x_list.size());
  std::vector<double> gt_y(x_list.size());
  std::vector<double> gt_yaw(x_list.size());

  std::vector<double> x_axis(x_list.size());
  
  for(int i = 0; i < x_list.size(); i++)
  {
    model_x[i] = CppAD::Value(x_list[i][4]);
    model_y[i] = CppAD::Value(x_list[i][5]);
    model_z[i] = CppAD::Value(x_list[i][6]);
    model_yaw[i] = CppAD::Value(2*CppAD::atan(x_list[i][2] / x_list[i][3]));
    
    gt_x[i] = traj[i].x;
    gt_y[i] = traj[i].y;
    gt_yaw[i] = traj[i].yaw;

    x_axis[i] = (double)i;
  }

  plt::subplot(1,3,1);
  plt::title("X-Y plot");
  plt::xlabel("[m]");
  plt::ylabel("[m]");
  plt::plot(model_x, model_y, "r", {{"label", "model"}});
  plt::plot(gt_x, gt_y, "b", {{"label", "gt"}});
  plt::legend();
  
  plt::subplot(1,3,2);
  plt::title("Time vs Elevation");
  plt::xlabel("Time [s]");
  plt::ylabel("Elevation [m]");
  plt::plot(model_z);

  plt::subplot(1,3,3);
  plt::title("Time vs yaw");
  plt::xlabel("Time [s]");
  plt::ylabel("yaw [Rads]");
  plt::plot(x_axis, model_yaw, "r", {{"label", "model"}});
  plt::plot(x_axis, gt_yaw, "b", {{"label", "gt"}});
  plt::legend();
  
  plt::show();
}

void Trainer::evaluateTrajectory(const std::vector<DataRow> &traj, std::vector<VectorAD> &x_list, double &loss)
{
  m_system_adf->setParams(m_params);
  
  m_system_adf->setNumSteps(m_eval_steps);
  
  VectorAD xk(m_system_adf->getStateDim());
  VectorAD xk1(m_system_adf->getStateDim());
  
  initializeState(traj[0], xk);
  x_list[0] = xk;

  ADF traj_len = 0;
  VectorAD gt_vec;
  for(int i = 1; i < x_list.size(); i++)
  {
    xk[HybridDynamics::STATE_DIM+0] = traj[i-1].vl;
    xk[HybridDynamics::STATE_DIM+1] = traj[i-1].vr;
    
    m_system_adf->integrate(xk, xk1);
    xk = xk1;
    x_list[i] = xk;
    
    gt_vec = VectorAD::Zero(m_system_adf->getStateDim());
    gt_vec[4] = ADF(traj[i].x);
    gt_vec[5] = ADF(traj[i].y);

    ADF dx = traj[i].x - traj[i-1].x;
    ADF dy = traj[i].y - traj[i-1].y;

    traj_len += CppAD::sqrt(dx*dx + dy*dy);
  }
  
  //plotTrajectory(traj, x_list);
  
  // This could also be a running loss instead of a terminal loss
  loss = CppAD::Value(m_system_adf->loss(gt_vec, x_list.back()) / traj_len);
  loss = std::sqrt(loss);
}

void Trainer::trainTrajectory(const std::vector<DataRow> &traj, std::vector<VectorAD> &x_list, VectorF &gradient, double& loss)
{
  VectorAD loss_ad(1);
  VectorAD xk(m_system_adf->getStateDim());
  VectorAD xk1(m_system_adf->getStateDim());
  
  initializeState(traj[0], xk);
  xk[HybridDynamics::STATE_DIM+0] = traj[0].vl;
  xk[HybridDynamics::STATE_DIM+1] = traj[0].vr;
  
  CppAD::Independent(m_params);
  m_system_adf->setParams(m_params);
    
  initializeState(traj[0], xk);
  x_list[0] = xk;
  loss_ad[0] = 0;
  
  VectorAD gt_vec;
  for(int i = 1; i < x_list.size(); i++)
  {
    xk[HybridDynamics::STATE_DIM+0] = traj[i-1].vl;
    xk[HybridDynamics::STATE_DIM+1] = traj[i-1].vr;
    
    m_system_adf->integrate(xk, xk1);
    xk = xk1;
    x_list[i] = xk;
    
    gt_vec = VectorAD::Zero(m_system_adf->getStateDim());
    gt_vec[4] = ADF(traj[i].x);
    gt_vec[5] = ADF(traj[i].y);
    
    loss_ad[0] += m_system_adf->loss(gt_vec, x_list[i]);
  }

  loss_ad[0] /= x_list.size();
  
  //loss_ad[0] = m_system_adf->loss(gt_vec, x_list.back());
  CppAD::ADFun<double> func(m_params, loss_ad);
  
  VectorF y0(1);
  y0[0] = 1;
  
  gradient = func.Reverse(1, y0);
  loss = CppAD::Value(loss_ad[0]);
}

void Trainer::initializeState(const DataRow &gt_state, VectorAD &xk_robot)
{
  ADF xk[m_system_adf->m_hybrid_dynamics.STATE_DIM];
  ADF xk_base[m_system_adf->m_hybrid_dynamics.STATE_DIM];
  
  VectorAD yaw_quat(4);
  
  yaw_quat[0] = 0;
  yaw_quat[1] = 0;
  yaw_quat[2] = std::sin(gt_state.yaw / 2.0); // rotating by yaw around z axis
  yaw_quat[3] = std::cos(gt_state.yaw / 2.0); // https://stackoverflow.com/questions/4436764/rotating-a-quaternion-on-1-axis
  
  // m_quat_stable * yaw_quat
  xk[0] = m_quat_stable[3]*yaw_quat[0] + m_quat_stable[0]*yaw_quat[3] + m_quat_stable[1]*yaw_quat[2] - m_quat_stable[2]*yaw_quat[1];
  xk[1] = m_quat_stable[3]*yaw_quat[1] + m_quat_stable[1]*yaw_quat[3] + m_quat_stable[2]*yaw_quat[0] - m_quat_stable[0]*yaw_quat[2];
  xk[2] = m_quat_stable[3]*yaw_quat[2] + m_quat_stable[2]*yaw_quat[3] + m_quat_stable[0]*yaw_quat[1] - m_quat_stable[1]*yaw_quat[0];
  xk[3] = m_quat_stable[3]*yaw_quat[3] - m_quat_stable[0]*yaw_quat[0] - m_quat_stable[1]*yaw_quat[1] - m_quat_stable[2]*yaw_quat[2];
  
  xk[4] = gt_state.x; //position
  xk[5] = gt_state.y;
  xk[6] = m_z_stable;

  xk[7] = 0; // Joint positions
  xk[8] = 0;
  xk[9] = 0;
  xk[10] = 0;

  xk[11] = 0; // Spatial Velocity
  xk[12] = 0;
  xk[13] = gt_state.wz;
  xk[14] = gt_state.vx;
  xk[15] = gt_state.vx;
  xk[16] = 0;

  xk[17] = 0; // Joint velocities
  xk[18] = 0;
  xk[19] = 0;
  xk[20] = 0;

  // Unfortunately, the state vector is not expressed at the COM. Depressing. So we must transform it
  m_system_adf->m_hybrid_dynamics.initStateCOM(&xk[0], &xk_base[0]);
  
  for(int i = 0; i < m_system_adf->m_hybrid_dynamics.STATE_DIM; i++)
  {
    xk_robot[i] = xk_base[i];
  }
  
  xk_robot[21] = gt_state.vl; // Control tire velocities
  xk_robot[22] = gt_state.vr;
}

void Trainer::save()
{
  m_system_adf->getParams(m_params);
  saveVec(m_params, m_param_file);
}
bool Trainer::saveVec(const VectorAD &params, const std::string &file_name)
{
  std::ofstream data_file(file_name);
  if(!data_file.is_open())
  {
    return true;
  }
  
  for(int i = 0; i < params.size(); i++)
  {
    data_file << CppAD::Value(params[i]) << ",";
  }
  data_file << "\n";

  return false;
}

void Trainer::load()
{
  if(!loadVec(m_params, m_param_file))
  {
    m_system_adf->setParams(m_params);
  }
}
bool Trainer::loadVec(VectorAD &params, const std::string &file_name)
{
  char comma;
  std::ifstream data_file(file_name);
  if(!data_file.is_open())
  {
    return true;
  }
  
  for(int i = 0; i < params.size(); i++)
  {
    data_file >> params[i];
    data_file >> comma;
  }
  
  return false;
}
