#include "Trainer.h"

#include <matplotlibcpp.h>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <cmath>


namespace plt = matplotlibcpp;


Trainer::Trainer()
{
  m_system_adf = std::make_shared<VehicleSystem<ADF>>();
 
  m_params = VectorAD::Zero(m_system_adf->getNumParams());
  m_system_adf->getDefaultParams(m_params);

  computeEqState();
  m_cnt = 0;
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

  std::cout << "Equillibrium State:\n";
  std::cout << "| z stable: " << m_z_stable << "\n";
  std::cout << "| quat[0]: " << m_quat_stable[0] << "\n";
  std::cout << "| quat[1]: " << m_quat_stable[1] << "\n";
  std::cout << "| quat[2]: " << m_quat_stable[2] << "\n";
  std::cout << "| quat[3]: " << m_quat_stable[3] << "\n";
}

// ,time,vel_left,vel_right,x,y,yaw,wx,wy,wz
void Trainer::loadDataFile(std::string fn)
{
  std::cout << "Opening " << fn << "\n";
  std::ifstream data_file(fn);

  if(!data_file.is_open())
  {
    std::cout << "File was not open\n";
    exit(0);
  }
  
  std::string line;
  std::getline(data_file, line); //ignore column heading
  
  char comma;
  int idx;
  float wx, wy;
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
  VectorF traj_grad(m_system_adf->getNumParams());
  int traj_len = m_system_adf->getNumSteps();
  float loss;
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
      
      std::cout << "Training on a trajectory: " << m_cnt << "\n";
      
      trainTrajectory(traj, x_list, traj_grad, loss);
      updateParams(traj_grad);
      plotTrajectory(traj, x_list);
      
      std::cout << "Loss: " << loss << "\n";
      m_cnt++;
    }
  }
  
}

void Trainer::evaluate()
{
  std::vector<VectorAD> x_list(m_system_adf->getNumSteps());
  int traj_len = m_system_adf->getNumSteps();
  char fn_array[100];
  
  float loss_avg = 0;
  float loss = 0;
  int cnt = 0;
  
  for(int i = 55; i <= 55; i++)
  {
    memset(fn_array, 0, 100);
    sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/CV3_data%02d.csv", i);
    
    std::string fn(fn_array);
    loadDataFile(fn);
    
    for(int j = 0; j < (m_data.size() - traj_len); j += traj_len)
    {
      std::vector<DataRow> traj(m_data.begin()+j, m_data.begin()+j+traj_len);
      evaluateTrajectory(traj, x_list, loss);
      plotTrajectory(traj, x_list);
      
      loss_avg += loss;
      cnt++;
    }
  }
  
  std::cout << "CV3 avg loss: " << loss_avg/cnt << "\n";

  loss_avg = 0;
  cnt = 0;
  
  memset(fn_array, 0, 100);
  sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/LD3_data%02d.csv", 1);
  std::string fn(fn_array);

  loadDataFile(fn);
  for(int j = 0; j < (m_data.size() - traj_len); j += traj_len)
  {
    std::vector<DataRow> traj(m_data.begin()+j, m_data.begin()+j+traj_len);
    evaluateTrajectory(traj, x_list, loss);
    plotTrajectory(traj, x_list);
    
    loss_avg += loss;
    cnt++;
  }
  
  std::cout << "LD3 avg loss: " << loss_avg/cnt << "\n";
}

void Trainer::updateParams(const VectorF &grad)
{
  for(int i = 0; i < m_params.size(); i++)
  {
    m_params[i] -= m_system_adf->getLearningRate()*ADF(grad[i]);
  }
}

void Trainer::plotTrajectory(const std::vector<DataRow> &traj, const std::vector<VectorAD> &x_list)
{
  assert(traj.size() == x_list.size());
  std::vector<float> model_x(x_list.size());
  std::vector<float> model_y(x_list.size());
  std::vector<float> gt_x(x_list.size());
  std::vector<float> gt_y(x_list.size());
  
  for(int i = 0; i < x_list.size(); i++)
  {
    model_x[i] = CppAD::Value(x_list[i][4]);
    model_y[i] = CppAD::Value(x_list[i][5]);
    
    gt_x[i] = traj[i].x;
    gt_y[i] = traj[i].y;
  }
  
  plt::plot(model_x, model_y, "r", {{"label", "model"}});
  plt::plot(gt_x, gt_y, "b", {{"label", "gt"}});
  plt::legend();
  plt::title("Trajectory Comparison");
  plt::show();
}

void Trainer::evaluateTrajectory(const std::vector<DataRow> &traj, std::vector<VectorAD> &x_list, float &loss)
{
  m_system_adf->setParams(m_params);
  
  loss = 0;
  VectorAD xk(m_system_adf->getStateDim());
  VectorAD xk1(m_system_adf->getStateDim());
  
  initializeState(traj[0], xk);
  x_list[0] = xk;
  
  for(int i = 1; i < x_list.size(); i++)
  {
    m_system_adf->integrate(xk, xk1);
    xk = xk1;
    x_list[i] = xk;
    
    VectorAD gt_vec = VectorAD::Zero(m_system_adf->getStateDim());
    gt_vec[4] = ADF(traj[i].x);
    gt_vec[5] = ADF(traj[i].y);
    
    loss += CppAD::Value(m_system_adf->loss(gt_vec, x_list[i]));
  }  
}

void Trainer::trainTrajectory(const std::vector<DataRow> &traj, std::vector<VectorAD> &x_list, VectorF &gradient, float& loss)
{
  CppAD::Independent(m_params);
  m_system_adf->setParams(m_params);
  
  VectorAD loss_ad(1);
  VectorAD xk(m_system_adf->getStateDim());
  VectorAD xk1(m_system_adf->getStateDim());
  
  initializeState(traj[0], xk);
  x_list[0] = xk;
  
  loss_ad[0] = 0;
  for(int i = 1; i < x_list.size(); i++)
  {
    m_system_adf->integrate(xk, xk1);
    xk = xk1;
    x_list[i] = xk;
    
    VectorAD gt_vec = VectorAD::Zero(m_system_adf->getStateDim());
    gt_vec[4] = ADF(traj[i].x);
    gt_vec[5] = ADF(traj[i].y);
    
    loss_ad[0] += m_system_adf->loss(gt_vec, x_list[i]);
  }
  
  CppAD::ADFun<float> func(m_params, loss_ad);
  
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
  VectorAD init_quat(4);
  
  yaw_quat[0] = 0;
  yaw_quat[1] = 0;
  yaw_quat[2] = std::sin(gt_state.yaw / 2.0); // rotating by yaw around z axis
  yaw_quat[3] = std::cos(gt_state.yaw / 2.0); // https://stackoverflow.com/questions/4436764/rotating-a-quaternion-on-1-axis
  
  // m_quat_stable * yaw_quat
  init_quat[0] = m_quat_stable[3]*yaw_quat[0] + m_quat_stable[0]*yaw_quat[3] + m_quat_stable[1]*yaw_quat[2] - m_quat_stable[2]*yaw_quat[1];
  init_quat[1] = m_quat_stable[3]*yaw_quat[1] + m_quat_stable[1]*yaw_quat[3] + m_quat_stable[2]*yaw_quat[0] - m_quat_stable[0]*yaw_quat[2];
  init_quat[2] = m_quat_stable[3]*yaw_quat[2] + m_quat_stable[2]*yaw_quat[3] + m_quat_stable[0]*yaw_quat[1] - m_quat_stable[1]*yaw_quat[0];
  init_quat[3] = m_quat_stable[3]*yaw_quat[3] - m_quat_stable[0]*yaw_quat[0] - m_quat_stable[1]*yaw_quat[1] - m_quat_stable[2]*yaw_quat[2];
  
  // xk[0] = init_quat[0]; //quaternion
  // xk[1] = init_quat[1];
  // xk[2] = init_quat[2];
  // xk[3] = init_quat[3];

  xk[0] = 0;
  xk[1] = 0;
  xk[2] = 0;
  xk[3] = 1;

  
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
  
  m_system_adf->m_hybrid_dynamics.initStateCOM(&xk[0], &xk_base[0]);
  
  for(int i = 0; i < m_system_adf->m_hybrid_dynamics.STATE_DIM; i++)
  {
    xk_robot[i] = xk_base[i];
  }
  
  xk_robot[21] = gt_state.vl; // Control tire velocities
  xk_robot[22] = gt_state.vr;

}
