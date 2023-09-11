#include "linear/LinearTrainer.h"

#include <matplotlibcpp.h>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <cmath>
#include <chrono>
#include <mutex>
#include <stdlib.h>

namespace plt = matplotlibcpp;


LinearTrainer::LinearTrainer()
{  
	m_system_adf = std::make_shared<LinearSystem>();  
	m_params = VectorAD::Zero(m_system_adf->getNumParams());
	m_batch_grad = VectorF::Zero(m_system_adf->getNumParams());
	m_squared_grad = VectorAD::Zero(m_system_adf->getNumParams());
	m_system_adf->getDefaultParams(m_params);
	
	m_cnt = 0;
	m_param_file = "/home/justin/linear.net";
}

LinearTrainer::~LinearTrainer()
{

}

// ,time,vel_left,vel_right,x,y,yaw,wx,wy,wz
void LinearTrainer::loadDataFile(std::string fn)
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

void LinearTrainer::train()
{
	double loss;
	double avg_loss = 0;
	char fn_array[100];
  
	for(int i = 1; i <= 17; i++)
	{
		memset(fn_array, 0, 100);
		sprintf(fn_array, "/home/justin/code/auvsl_dynamics_bptt/scripts/Train3_data%02d.csv", i);
		
		std::string fn(fn_array);
		loadDataFile(fn);
		
		for(int j = 0; j < m_data.size(); j++)
		{
			VectorAD xk(m_system_adf->getStateDim());
			initializeState(m_data[j], xk);
			
			VectorAD u(2);
			VectorAD xd(3);
			VectorAD gt_vec(3);
			u[0] = m_data[j].vl;
			u[1] = m_data[j].vr;
			gt_vec[0] = m_data[j].vx;
			gt_vec[1] = m_data[j].vy;
			gt_vec[2] = m_data[j].wz;

			VectorAD params(m_system_adf->getNumParams());
			params = m_params;
			
			CppAD::Independent(params);
			
			m_system_adf->setParams(params);
			m_system_adf->forward(u, xd);
			VectorAD loss_vec(1);
			loss_vec[0] = m_system_adf->loss(gt_vec, xd);
			
			CppAD::ADFun<double> func(params, loss_vec);

			VectorF y0(1);
			y0[0] = 1;
			VectorF gradient(m_system_adf->getNumParams());
			gradient = func.Reverse(1, y0);
			
			loss = CppAD::Value(loss_vec[0]);
			
			m_batch_grad += gradient;
			avg_loss += loss;
			m_cnt++;
		}		
	}
	
	std::cout << "Avg Loss: " << avg_loss / m_cnt << ", Batch Grad[0]: " << m_batch_grad[0] << "\n";
	std::flush(std::cout);
	updateParams(m_batch_grad / m_cnt);
	m_cnt = 0;
	avg_loss = 0;
	
	save();
}

void LinearTrainer::evaluate_train3()
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
			std::vector<DataRow> traj(m_data.begin()+j, m_data.begin()+j+m_eval_steps);
			evaluateTrajectory(traj, x_list, loss);
			// plotTrajectory(traj, x_list);
		
			loss_avg += loss;
			cnt++;
		}
	}
  
	std::cout << "Train3 avg loss: " << loss_avg/cnt << "\n";
}


void LinearTrainer::evaluate_cv3()
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
			std::vector<DataRow> traj(m_data.begin()+j, m_data.begin()+j+m_eval_steps);
			evaluateTrajectory(traj, x_list, loss);
			//plotTrajectory(traj, x_list);
      
			loss_avg += loss;
			cnt++;
		}
	}
  
	std::cout << "CV3 avg loss: " << loss_avg/cnt << "\n";
}

void LinearTrainer::evaluate_ld3()
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
		std::vector<DataRow> traj(m_data.begin()+j, m_data.begin()+j+m_eval_steps);
		evaluateTrajectory(traj, x_list, loss);
		// plotTrajectory(traj, x_list);
    
		loss_avg += loss;
		cnt++;
	}
  
	std::cout << "LD3 avg loss: " << loss_avg/cnt << "\n";
}

void LinearTrainer::updateParams(const VectorF &grad)
{
	ADF norm = 0;
	float grad_idx;
  
	for(int i = 0; i < m_params.size(); i++)
	{
		grad_idx = grad[i];
	
		m_squared_grad[i] = 0.9*m_squared_grad[i] + 0.1*ADF(grad_idx*grad_idx);
		m_params[i] -= (m_lr/(CppAD::sqrt(m_squared_grad[i]) + 1e-6))*ADF(grad_idx);
		norm += CppAD::abs(m_params[i]);
	}

	ADF update0 = (m_lr/(CppAD::sqrt(m_squared_grad[0]) + 1e-6))*ADF(grad[0]);
	std::cout << "Param norm: " << CppAD::Value(norm)
			  << " Param[0]: " << CppAD::Value(m_params[0])
			  << " dParams[0]: " << CppAD::Value(update0) << "\n";
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

void LinearTrainer::plotTrajectory(const std::vector<DataRow> &traj, const std::vector<VectorAD> &x_list)
{
	assert(traj.size() == x_list.size());
	std::vector<double> model_x(x_list.size());
	std::vector<double> model_y(x_list.size());
	std::vector<double> model_yaw(x_list.size());  
	std::vector<double> gt_x(x_list.size());
	std::vector<double> gt_y(x_list.size());
	std::vector<double> gt_yaw(x_list.size());
	std::vector<double> gt_vl(x_list.size());
	std::vector<double> gt_vr(x_list.size());
	std::vector<double> x_axis(x_list.size());
  
	for(int i = 0; i < x_list.size(); i++)
	{
		model_x[i] = CppAD::Value(x_list[i][0]);
		model_y[i] = CppAD::Value(x_list[i][1]);
		model_yaw[i] = CppAD::Value(2*CppAD::atan(x_list[i][2] / x_list[i][3])); //this is an approximation
		gt_x[i] = traj[i].x;
		gt_y[i] = traj[i].y;
		gt_yaw[i] = traj[i].yaw;
		gt_vl[i] = traj[i].vl;
		gt_vr[i] = traj[i].vr;
		x_axis[i] = .05*i;
	}

	std::vector<double> aspect_ratio_hack_x(2);
	std::vector<double> aspect_ratio_hack_y(2);
	double min = std::min(*std::min_element(model_x.begin(), model_x.end()), *std::min_element(model_y.begin(), model_y.end()));
	double max = std::max(*std::max_element(model_x.begin(), model_x.end()), *std::max_element(model_y.begin(), model_y.end()));
	aspect_ratio_hack_x[0] = min;
	aspect_ratio_hack_y[0] = min;
	aspect_ratio_hack_x[1] = max;
	aspect_ratio_hack_y[1] = max;

	plt::subplot(1,2,1);
	plt::title("X-Y plot");
	plt::xlabel("[m]");
	plt::ylabel("[m]");
	plt::plot(model_x, model_y, "r", {{"label", "model"}});
	plt::plot(gt_x, gt_y, "b", {{"label", "gt"}});
	plt::scatter(aspect_ratio_hack_x, aspect_ratio_hack_y);
	plt::legend();
  
	plt::subplot(1,2,2);
	plt::title("Time vs yaw");
	plt::xlabel("Time [s]");
	plt::ylabel("yaw [Rads]");
	plt::plot(x_axis, model_yaw, "r", {{"label", "model"}});
	plt::plot(x_axis, gt_yaw, "b", {{"label", "gt"}});
	plt::legend();
	
	plt::show();
}

void LinearTrainer::evaluateTrajectory(const std::vector<DataRow> &traj, std::vector<VectorAD> &x_list, double &loss)
{
	m_system_adf->setParams(m_params);
  
	VectorAD xk(m_system_adf->getStateDim());
	VectorAD xk1(m_system_adf->getStateDim());
  
	initializeState(traj[0], xk);
	x_list[0] = xk;

	ADF traj_len = 0;
	VectorAD gt_vec;
	for(int i = 1; i < x_list.size(); i++)
	{
		xk[3] = traj[i-1].vl;
		xk[4] = traj[i-1].vr;
    
		m_system_adf->integrate(xk, xk1);
		xk = xk1;
		x_list[i] = xk;
	  
		gt_vec = VectorAD::Zero(m_system_adf->getStateDim());
	  
		gt_vec[0] = ADF(traj[i].x);
		gt_vec[1] = ADF(traj[i].y);
		gt_vec[2] = ADF(traj[i].yaw);
	  
		ADF dx = traj[i].x - traj[i-1].x;
		ADF dy = traj[i].y - traj[i-1].y;
	  
		traj_len += CppAD::sqrt(dx*dx + dy*dy);
	}
  
	// plotTrajectory(traj, x_list);
  
	// This could also be a running loss instead of a terminal loss
	ADF ang_mse;
	ADF lin_mse;
	m_system_adf->evaluate(gt_vec, x_list.back(), ang_mse, lin_mse);

	// std::cout << "Lin err: " << CppAD::Value(CppAD::sqrt(lin_mse)) << " Lin Displacement: " << CppAD::Value(traj_len) << "\n";
	// std::cout << "Lin err " << CppAD::Value(CppAD::sqrt(lin_mse)) << " Ang err " << CppAD::Value(CppAD::sqrt(ang_mse)) << "\n";

	loss = CppAD::Value(CppAD::sqrt(lin_mse) / traj_len);
	// std::cout << "Lin err: " << CppAD::Value(CppAD::sqrt(lin_mse))
	// 	    << " traj_len: " << CppAD::Value(traj_len)
	// 	    << " Relative Linear: " << loss << "\n";
}


void LinearTrainer::initializeState(const DataRow &gt_state, VectorAD &xk_robot)
{
	xk_robot[0] = gt_state.x;
	xk_robot[1] = gt_state.y;
	xk_robot[2] = gt_state.yaw;
	xk_robot[3] = gt_state.vl;
	xk_robot[3] = gt_state.vr;
}

void LinearTrainer::save()
{
  saveVec(m_params, m_param_file);
}
bool LinearTrainer::saveVec(const VectorAD &params, const std::string &file_name)
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

void LinearTrainer::load()
{
  if(!loadVec(m_params, m_param_file))
  {
    m_system_adf->setParams(m_params);
  }
}
bool LinearTrainer::loadVec(VectorAD &params, const std::string &file_name)
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
