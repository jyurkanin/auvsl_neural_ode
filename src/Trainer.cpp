#include "Trainer.h"

#include <matplotlibcpp.h>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <cmath>
#include <thread>
#include <chrono>
#include <mutex>
#include <stdlib.h>

namespace plt = matplotlibcpp;


bool g_parallel_mode;
bool give_me_parallel_mode()
{
  return g_parallel_mode;
}


std::vector<std::thread::id> g_id_map;
void put_id_in_map(int id_num)
{
  g_id_map[id_num] = std::this_thread::get_id();
}


size_t give_me_thread_id()
{
  for(int i = 0; i < g_id_map.size(); i++)
  {
    if(g_id_map[i] == std::this_thread::get_id())
    {
      return (size_t) i;
    }
  }

  std::cout << "No bueno aqui!\n";
  return 100; //bad.
}

Trainer::Trainer(std::shared_ptr<SystemFactory<ADF>> factory, int num_threads) : m_factory_adf{factory}, m_num_threads{num_threads}
{
  // Enable CppAD multithreading. Sucky.
  g_id_map.resize(num_threads+1);
  put_id_in_map(num_threads);

  g_parallel_mode = false;
  CppAD::thread_alloc::parallel_setup(m_num_threads+1, give_me_parallel_mode, give_me_thread_id);
  
  CppAD::parallel_ad<ADF>();
  g_parallel_mode = true; // "O.K." - Saitama
  
  m_system_adf = m_factory_adf->makeSystem();
  m_params = VectorAD::Zero(m_system_adf->getNumParams());
  m_batch_grad = VectorF::Zero(m_system_adf->getNumParams());
  m_squared_grad = VectorAD::Ones(m_system_adf->getNumParams());
  
  m_system_adf->getDefaultParams(m_params);
  m_system_adf->setParams(m_params);
  
  computeEqState();
  
  m_cnt = 0;
  m_param_file = "/home/justin/tire.net";

  m_best_CV3 = .096; //large number
}

Trainer::~Trainer()
{

}

void Trainer::computeEqState()
{
	VectorAD state = VectorAD::Zero(m_system_adf->getStateDim());

	m_system_adf->getDefaultInitialState(state);
	
	m_quat_stable = VectorAD::Zero(4);
	m_quat_stable[0] = state[0];
	m_quat_stable[1] = state[1];
	m_quat_stable[2] = state[2];
	m_quat_stable[3] = state[3];
	
	m_z_stable = state[6];
	std::cout << "Stable Quaternion: "
			  << m_quat_stable[0] << ", "
		      << m_quat_stable[1] << ", "
		      << m_quat_stable[2] << ", "
		      << m_quat_stable[3] << "\n";
		
	std::cout << "z_stable: " << m_z_stable << "\n";
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
	GroundTruthDataRow row;

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
		row.z = CppAD::Value(m_z_stable);
		
		m_data.push_back(row);
	}

}

void Trainer::train()
{
	VectorF traj_grad(m_system_adf->getNumParams());
	double loss;
	double avg_loss = 0;
	std::vector<VectorAD> x_list(m_train_steps);
	char fn_array[100];

	int cnt_actual = 0;
	
	for(int i = 1; i <= 17; i++)
	{
		memset(fn_array, 0, 100);
		sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/Train3_data%02d.csv", i);
    
		std::string fn(fn_array);
		loadDataFile(fn);
    
		for(int j = 0; j < (m_data.size() - m_train_steps); j += m_inc_train_steps)
		{
			std::vector<GroundTruthDataRow> traj(m_data.begin() + j, m_data.begin() + j + m_train_steps);
      
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
				// plotTrajectory(traj, x_list);
				std::cout << "Loss: " << loss << "\tdParams: " << traj_grad[0] << "\n";
				std::flush(std::cout);
				m_cnt++;
				cnt_actual++;

				std::flush(std::cout);
				updateParams(m_batch_grad / m_cnt);
				m_cnt = 0;
			}      
		}
				
		save();
	}

	std::cout << "Average Loss: " << avg_loss / cnt_actual << "\n";
	avg_loss = 0;
}

void Trainer::trainThreads()
{
  auto worker_lambda = [](Trainer::Worker *worker)
  {
    worker->work();
  };
  
  m_workers.clear();
  m_workers.resize(m_num_threads);
    
  for(int i = 0; i < m_workers.size(); i++)
  {
    m_workers[i] = Worker(this);
    m_workers[i].m_idle.store(true);
    m_workers[i].m_ready.store(false);
    m_workers[i].m_waiting.store(false);
    
    m_workers[i].m_keep_alive.store(true);
    
    m_workers[i].m_id = i;
    m_workers[i].m_thread = std::thread(worker_lambda, &m_workers[i]);
  }
  
  m_cnt = 0;
  m_batch_loss = 0;
  m_batch_grad = VectorF::Zero(m_system_adf->getNumParams());
  
  int m_train_steps = m_train_steps;
  char fn_array[100];

  int cnt_workers = 0;
  for(int i = 1; i <= 16; i++)
  {
    memset(fn_array, 0, 100); //todo: uncomment
    sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/Train3_data%02d.csv", i);
	
    std::string fn(fn_array);
    loadDataFile(fn);
	
    for(int j = 0; j < (m_data.size() - m_train_steps); j += m_inc_train_steps)
    {
      std::vector<GroundTruthDataRow> traj(m_data.begin() + j, m_data.begin() + j + m_train_steps);
      assignWork(traj);
    }
  }
  
  finishWork();
  
  std::cout << "Avg Loss: " << m_batch_loss / m_cnt << "\n";
  std::flush(std::cout);
  save();
  updateParams(m_batch_grad / m_cnt);

  if((m_batch_loss/m_cnt) < m_best_CV3)
  {
	  m_best_CV3 = m_batch_loss/m_cnt;
	  std::cout << "dabest " << m_best_CV3 << "\n";
	  saveVec(m_params, std::string("/home/justin/best_tire.net"));
  }
}

void Trainer::assignWork(const std::vector<GroundTruthDataRow> &traj)
{
	while(true)
	{
		for(int i = 0; i < m_workers.size(); i++)
		{
			if(m_workers[i].m_idle.load())
			{
				// std::cout << "Was Idle\n"; std::flush(std::cout);
				m_workers[i].m_traj = traj;
				m_workers[i].m_idle.store(false);
				m_workers[i].m_ready.store(true);
				return;
			}
			else if(m_workers[i].m_waiting.load())
			{
				// std::cout << "Was Waiting\n"; std::flush(std::cout);
				combineResults(m_batch_grad, m_workers[i].m_grad,
							   m_batch_loss, m_workers[i].m_loss);
	
				m_workers[i].m_waiting.store(false);
				m_workers[i].m_idle.store(true);
			}
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}  
}

void Trainer::finishWork()
{
  for(int i = 0; i < m_workers.size(); i++)
  {
    if(m_workers[i].m_thread.joinable())
    {
      m_workers[i].m_keep_alive.store(false);
      m_workers[i].m_thread.join();
      combineResults(m_batch_grad, m_workers[i].m_grad, m_batch_loss, m_workers[i].m_loss);
    }
  }
}

void Trainer::evaluate_validation_dataset()
{
  std::vector<VectorAD> x_list(m_eval_steps);
  char fn_array[100];
  
  double loss = 0;
  int cnt = 0;
  
  int i = 17;

  memset(fn_array, 0, 100); //todo: uncomment
  sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/Train3_data%02d.csv", i);
  
  std::string fn(fn_array);
  loadDataFile(fn);
    
  for(int j = 0; j < (m_data.size() - m_eval_steps); j += m_inc_eval_steps)
  {
	  std::vector<GroundTruthDataRow> traj(m_data.begin()+j, m_data.begin()+j+m_eval_steps);
	  evaluateTrajectory(traj, x_list, loss);
	  // plotTrajectory(traj, x_list);
  }
  
  std::cout << "Train3_17 validation loss: " << loss << "\n";	
}

void Trainer::evaluate_train3()
{
  std::vector<VectorAD> x_list(m_eval_steps);
  char fn_array[100];
  
  double loss_avg = 0;
  double loss = 0;
  int cnt = 0;
  
  for(int i = 1; i <= 17; i++)
  {
    memset(fn_array, 0, 100); //todo: uncomment
    sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/Train3_data%02d.csv", i);
	
    // memset(fn_array, 0, 100);
    // sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/CV3_data%02d.csv", i);
	
    std::string fn(fn_array);
    loadDataFile(fn);
    
    for(int j = 0; j < (m_data.size() - m_eval_steps); j += m_inc_eval_steps)
    {
		std::vector<GroundTruthDataRow> traj(m_data.begin()+j, m_data.begin()+j+m_eval_steps);
		evaluateTrajectory(traj, x_list, loss);
		// plotTrajectory(traj, x_list);
		
		loss_avg += loss;
		cnt++;
    }
  }
  
  std::cout << "Train3 avg loss: " << loss_avg/cnt << "\n";
}


void Trainer::evaluate_cv3()
{
	std::vector<VectorAD> x_list(m_eval_steps);
	char fn_array[100];
  
	double loss_avg = 0;
	double loss = 0;
	int cnt = 0;
  
	for(int i = 1; i <= 144; i++)
	{
		memset(fn_array, 0, 100);
		sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/CV3_data%02d.csv", i);
    
		std::string fn(fn_array);
		loadDataFile(fn);
    
		for(int j = 0; j < (m_data.size() - m_eval_steps); j += m_inc_eval_steps)
		{
			std::vector<GroundTruthDataRow> traj(m_data.begin()+j, m_data.begin()+j+m_eval_steps);
			evaluateTrajectory(traj, x_list, loss);
			// plotTrajectory(traj, x_list);
      
			loss_avg += loss;
			cnt++;
		}
	}
  
	std::cout << "CV3 avg loss: " << loss_avg/cnt << "\n";
}

void Trainer::evaluate_ld3()
{
	std::vector<VectorAD> x_list(m_eval_steps);
	char fn_array[100];
  
	double loss_avg = 0;
	double loss = 0;
	int cnt = 0;
    
	memset(fn_array, 0, 100);
	sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/LD3_data%02d.csv", 1);
	std::string fn(fn_array);

	loadDataFile(fn);
	for(int j = 0; j < (m_data.size() - m_eval_steps); j += m_inc_eval_steps)
	{
		std::vector<GroundTruthDataRow> traj(m_data.begin()+j, m_data.begin()+j+m_eval_steps);
		evaluateTrajectory(traj, x_list, loss);
		// plotTrajectory(traj, x_list);
    
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
	
		m_squared_grad[i] = 0.95*m_squared_grad[i] + 0.05*ADF(grad_idx*grad_idx);
		m_params[i] -= (m_lr/(CppAD::sqrt(m_squared_grad[i]) + 1e-6))*ADF(grad_idx);
		m_params[i] -= m_lr*m_l1_weight*m_params[i];
		// m_params[i] -= m_lr*ADF(grad_idx); // Vanilla gradient descent
		norm += CppAD::abs(grad[i]);
	}

	ADF update0 = (m_lr/(CppAD::sqrt(m_squared_grad[0]) + 1e-6))*ADF(grad[0]);
	//ADF update0 = m_lr*ADF(grad[0]);
	std::cout << "Gradient norm: " << CppAD::Value(norm)
			  << " Param[0]: " << CppAD::Value(m_params[0])
			  << " grad[0]: " << grad[0]
			  << " update[0]: " << CppAD::Value(update0)
			  << " squared_grad[0]: " << CppAD::Value(CppAD::sqrt(m_squared_grad[0])) << "\n";
	
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

void Trainer::plotTrajectory(const std::vector<GroundTruthDataRow> &traj, const std::vector<VectorAD> &x_list)
{
  assert(traj.size() == x_list.size());
  std::vector<double> model_x(x_list.size());
  std::vector<double> model_y(x_list.size());
  std::vector<double> model_z(x_list.size());
  std::vector<double> model_yaw(x_list.size());
  std::vector<double> model_vx(x_list.size());
  std::vector<double> model_vy(x_list.size());
  
  std::vector<double> gt_x(x_list.size());
  std::vector<double> gt_y(x_list.size());
  std::vector<double> gt_yaw(x_list.size());
  std::vector<double> gt_vx(x_list.size());
  std::vector<double> gt_vy(x_list.size());
  std::vector<double> gt_wz(x_list.size());
  std::vector<double> gt_vl(x_list.size());
  std::vector<double> gt_vr(x_list.size());
  
  std::vector<double> x_axis(x_list.size());
  
  for(int i = 0; i < x_list.size(); i++)
  {
    model_x[i] = CppAD::Value(x_list[i][4]);
    model_y[i] = CppAD::Value(x_list[i][5]);
    model_z[i] = CppAD::Value(x_list[i][6]);
    model_yaw[i] = CppAD::Value(2*CppAD::atan(x_list[i][2] / x_list[i][3])); //this is an approximation
	model_vx[i] = CppAD::Value(x_list[i][14]);
	model_vy[i] = CppAD::Value(x_list[i][15]);
	
    gt_x[i] = traj[i].x;
    gt_y[i] = traj[i].y;
    gt_yaw[i] = traj[i].yaw;
	gt_wz[i] = traj[i].wz;
	gt_vx[i] = traj[i].vx;
	gt_vy[i] = traj[i].vy;
	gt_vl[i] = traj[i].vl;
	gt_vr[i] = traj[i].vr;
	
    x_axis[i] = .01*i;
  }

  std::vector<double> aspect_ratio_hack_x(2);
  std::vector<double> aspect_ratio_hack_y(2);
  double min = std::min(*std::min_element(model_x.begin(), model_x.end()), *std::min_element(model_y.begin(), model_y.end()));
  double max = std::max(*std::max_element(model_x.begin(), model_x.end()), *std::max_element(model_y.begin(), model_y.end()));
  aspect_ratio_hack_x[0] = min;
  aspect_ratio_hack_y[0] = min;
  aspect_ratio_hack_x[1] = max;
  aspect_ratio_hack_y[1] = max;

  plt::subplot(1,6,1);
  plt::plot(gt_vl);
  plt::plot(gt_vr);
  plt::title("Vl Vr");
  
  plt::subplot(1,6,2);
  plt::title("X-Y plot");
  plt::xlabel("[m]");
  plt::ylabel("[m]");
  plt::plot(model_x, model_y, "r", {{"label", "model"}});
  plt::plot(gt_x, gt_y, "b", {{"label", "gt"}});
  plt::scatter(aspect_ratio_hack_x, aspect_ratio_hack_y);
  plt::legend();
  
  plt::subplot(1,6,3);
  plt::title("Time vs Elevation");
  plt::xlabel("Time [s]");
  plt::ylabel("Elevation [m]");
  plt::plot(model_z);

  plt::subplot(1,6,4);
  plt::title("Time vs yaw");
  plt::xlabel("Time [s]");
  plt::ylabel("yaw [Rads]");
  plt::plot(x_axis, model_yaw, "r", {{"label", "model"}});
  plt::plot(x_axis, gt_yaw, "b", {{"label", "gt"}});
  plt::legend();
  
  plt::subplot(1,6,5);
  plt::plot(gt_vx);
  plt::plot(model_vx);
  
  plt::subplot(1,6,6);
  plt::plot(gt_vy);
  plt::plot(model_vy);
  plt::show();
}

void Trainer::evaluateTrajectory(const std::vector<GroundTruthDataRow> &traj, std::vector<VectorAD> &x_list, double &loss)
{
	m_system_adf->setParams(m_params);
  
	VectorAD xk(m_system_adf->getStateDim() + m_system_adf->getControlDim());
	VectorAD xk1(m_system_adf->getStateDim() + m_system_adf->getControlDim());
  
	xk = m_system_adf->initializeState(traj[0]);
	x_list[0] = xk;

	ADF traj_len = 0;
	VectorAD gt_vec;
	for(int i = 1; i < x_list.size(); i++)
	{
		xk[m_system_adf->getStateDim()+0] = traj[i-1].vl;
		xk[m_system_adf->getStateDim()+1] = traj[i-1].vr;
    
		m_system_adf->integrate(xk, xk1);
		xk = xk1;
		x_list[i] = xk;
    
		gt_vec = VectorAD::Zero(m_system_adf->getStateDim());
    
		gt_vec[3] = ADF(traj[i].yaw);
		gt_vec[4] = ADF(traj[i].x);
		gt_vec[5] = ADF(traj[i].y);

		ADF dx = traj[i].x - traj[i-1].x;
		ADF dy = traj[i].y - traj[i-1].y;

		traj_len += CppAD::sqrt(dx*dx + dy*dy);
	}
  
	plotTrajectory(traj, x_list);
  
	// This could also be a running loss instead of a terminal loss
	ADF ang_mse;
	ADF lin_mse;
	m_system_adf->evaluate(gt_vec, x_list.back(), ang_mse, lin_mse);
	//std::cout << "Lin err " << CppAD::Value(CppAD::sqrt(lin_mse)) << " Ang err " << CppAD::Value(CppAD::sqrt(ang_mse)) << "\n";

	loss = CppAD::Value(CppAD::sqrt(lin_mse) / traj_len);
	// std::cout << "Lin err: " << CppAD::Value(CppAD::sqrt(lin_mse))
	// 	    << " traj_len: " << CppAD::Value(traj_len)
	// 	    << " Relative Linear: " << loss << "\n";
  
  
  
}

void Trainer::trainTrajectory(const std::vector<GroundTruthDataRow> &traj,
							  std::vector<VectorAD> &x_list,
							  VectorF &gradient,
							  double& loss)
{
	std::shared_ptr<System<ADF>> system_adf = m_factory_adf->makeSystem();
  
	VectorAD params = m_params;
	VectorAD loss_ad(1);
	VectorAD xk(m_system_adf->getStateDim() + m_system_adf->getControlDim());
	VectorAD xk1(m_system_adf->getStateDim() + m_system_adf->getControlDim());
  
	xk = m_system_adf->initializeState(traj[0]);
	xk[m_system_adf->getStateDim()+0] = traj[0].vl;
	xk[m_system_adf->getStateDim()+1] = traj[0].vr;
  
	CppAD::Independent(params);
	system_adf->setParams(params);
    
	xk = m_system_adf->initializeState(traj[0]);
	x_list[0] = xk;
	loss_ad[0] = 0;
  
	ADF traj_len = 0;
  
	VectorAD gt_vec;
	for(int i = 1; i < x_list.size(); i++)
	{
		xk[m_system_adf->getStateDim()+0] = traj[i-1].vl;
		xk[m_system_adf->getStateDim()+1] = traj[i-1].vr;
	
		system_adf->integrate(xk, xk1);
		xk = xk1;
		x_list[i] = xk;
    
		gt_vec = VectorAD::Zero(system_adf->getStateDim());
		gt_vec[3] = ADF(traj[i].yaw);
		gt_vec[4] = ADF(traj[i].x);
		gt_vec[5] = ADF(traj[i].y);
		gt_vec[13] = ADF(traj[i].wz);
		gt_vec[14] = ADF(traj[i].vx);
		gt_vec[15] = ADF(traj[i].vy);
    
		ADF dx = traj[i].x - traj[i-1].x;
		ADF dy = traj[i].y - traj[i-1].y;
    
		traj_len += CppAD::sqrt(dx*dx + dy*dy);
		loss_ad[0] += system_adf->loss(gt_vec, x_list[i]);
	}

	loss_ad[0] /= x_list.size();
	//loss_ad[0] = system_adf->loss(gt_vec, x_list.back());
  
	if(traj_len == 0)
	{
		traj_len = 1; //dont let a divide by zero happen.
	}
  
	loss_ad[0] = loss_ad[0] / traj_len;
	CppAD::ADFun<double> func(params, loss_ad);
  
	// std::cout << "Loss: " << CppAD::Value(system_adf->loss(gt_vec, x_list.back())) << "\n\n";
  
	VectorF y0(1);
	y0[0] = 1;
  
	gradient = func.Reverse(1, y0);
	loss = CppAD::Value(loss_ad[0]);
}

bool Trainer::combineResults(VectorF &batch_grad,
			     const VectorF &sample_grad,
			     double &batch_loss,
			     const double &sample_loss)
{
  bool has_explosion = false;
  for(int i = 0; i < m_params.size(); i++)
  {
    if(fabs(sample_grad[i]) > 100.0)
    {
      has_explosion = true;
      std::cout << "Explosion " << i << ":" << sample_grad[i] << "\n";
      break;
    }
  }
  
  if(!has_explosion)
  {
    batch_grad += sample_grad;
    batch_loss += sample_loss;
    m_cnt++;
    
    std::cout << "Loss: " << sample_loss << "\tdParams: " << sample_grad[0] << "\n";
    std::flush(std::cout);
    return true;
  }
  
  return false;
}

void Trainer::save()
{
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
	else
	{
		std::cout << "No param save file found\n";
		m_system_adf->getDefaultParams(m_params);
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





// Worker Definition
Trainer::Worker::Worker()
{
  
}

Trainer::Worker::~Worker()
{
  if(m_thread.joinable()) { m_thread.join(); }
}


Trainer::Worker::Worker(const Worker& other)
{
  m_idle = other.m_idle.load();
  m_ready = other.m_ready.load();
  m_waiting = other.m_waiting.load();
  m_keep_alive = other.m_keep_alive.load();
  
  m_grad = other.m_grad;
  m_loss = other.m_loss;
  m_trainer = other.m_trainer;
}

Trainer::Worker::Worker(Trainer* trainer) : m_trainer{trainer}
{
  
}

void Trainer::Worker::work()
{
  put_id_in_map(m_id);
  
  while(m_keep_alive)
  {
    if(m_ready.load())
    {
      // std::cout << "Was Ready\n"; std::flush(std::cout);
      m_ready.store(false);
      
      std::vector<VectorAD> x_list(m_trainer->m_train_steps);
      m_trainer->trainTrajectory(m_traj, x_list, m_grad, m_loss);
      
      m_waiting.store(true);
    }
    else
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

const Trainer::Worker& Trainer::Worker::operator=(const Trainer::Worker& other)
{
  this->m_idle = other.m_idle.load();
  this->m_ready = other.m_ready.load();
  this->m_waiting = other.m_waiting.load();
  this->m_keep_alive = other.m_keep_alive.load();
  
  this->m_grad = other.m_grad;
  this->m_loss = other.m_loss;
  this->m_trainer = other.m_trainer;
  
  return *this;
}
