#include <matplotlibcpp.h>

#include "gtest/gtest.h"
#include "TireNetwork.h"
#include "VehicleSystem.h"
#include "TestTerrainMaps.h"
#include "generated/model_constants.h"
#include "types/Scalars.h"

namespace plt = matplotlibcpp;

namespace{
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorF;
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixF;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, 1> VectorAD;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> MatrixAD;

  void plot_cross(float x1, float x2, float y1, float y2)
  {
	  std::vector<float> x(2);
	  std::vector<float> y(2);

	  x[0] = x1;
	  x[1] = x2;
	  y[0] = 0;
	  y[1] = 0;
	  plt::plot(x, y, "k");
	  x[0] = 0;
	  x[1] = 0;
	  y[0] = y1;
	  y[1] = y2;
	  plt::plot(x, y, "k");    
  }
  
  bool loadVec(VectorAD &params, const std::string &file_name)
  {
	  char comma;
	  std::ifstream data_file(file_name);
	  if(!data_file.is_open())
	  {
		  std::cout << "Failed to open file\n";
		  return true;
	  }
	  
	  for(int i = 0; i < params.size(); i++)
	  {
		  data_file >> params[i];
		  data_file >> comma;
	  }
	  
	  return false;
  }  
  
  TEST(TireNetwork, check_params)
  {
	  std::shared_ptr<const FlatTerrainMap<ADF>> map;
	  VehicleSystem<ADF> system1(map);
	  VehicleSystem<ADF> system2(map);
    
	  VectorAD params1(system1.getNumParams());
	  VectorAD params2(system1.getNumParams());
    
	  system1.getDefaultParams(params1);
	  system2.setParams(params1);
	  system2.getParams(params2);
    
	  for(int i = 0; i < params1.size(); i++)
	  {
		  EXPECT_EQ(params1[i], params2[i]);  
	  }
  }
  
  TEST(TireNetwork, vx_fx_plot)
  {
	  TireNetwork tire_network;
	  tire_network.load_model();
	  
	  VectorAD params = VectorAD::Zero(tire_network.getNumParams());
	  if(!loadVec(params, "/home/justin/tire.net"))
	  {
		  tire_network.setParams(params, 0);
	  }
	  
	  Eigen::Matrix<ADF,8,1> features;
	  Eigen::Matrix<ADF,TireNetwork::num_out_features,1> forces;
	  
	  features[0] = 0;
	  features[1] = 0;
	  features[2] = 0;
	  features[3] = 0.005;
    
	  features[4] = 0.0;
	  features[5] = 0.0;
	  features[6] = 0.0;
	  features[7] = 0.0;
    
	  int len = 1000;
	  int num = 11;
	  std::vector<float> vx_vec(len);
	  std::vector<float> fx_vec(len);
	  
	  for(int j = 0; j < (num+1); j++)
	  {
		  ADF tire_vx = 0.1*ADF((2.0*j/(float)num) - 1.0);
		  
		  for(int i = 0; i < len; i++)
		  {
			  ADF tire_tangent_vel = 0.2*ADF((2.0*i/(float)len) - 1.0);
			  
			  features[0] = tire_vx;
			  features[2] = tire_tangent_vel/.098;
			  
			  tire_network.forward(features, forces, 0);
			  
			  vx_vec[i] = CppAD::Value(tire_tangent_vel - tire_vx);
			  fx_vec[i] = CppAD::Value(forces[0]);
		  }

		  std::stringstream stream;
		  stream << std::fixed << std::setprecision(2) << CppAD::Value(tire_vx);
		  std::string label = stream.str();
		  
		  float grey = (0.8*j / num);
		  plt::plot(vx_vec, fx_vec, {{"color", std::to_string(grey)}, {"label", label}});
	  }
	  
	  plt::legend();
	  plt::xlabel("Velocity Difference (m/s)");
	  plt::ylabel("Longitudinal Force (N)");
	  plt::show();
	  
	  
	  for(int i = 0; i < len; i++)
	  {
		  ADF tire_vx = 0.0;
		  ADF tire_tangent_vel = 0.001*ADF((2.0*i/(float)len) - 1.0);
		  features[0] = tire_vx;
		  features[2] = tire_tangent_vel / 0.098;

		  tire_network.forward(features, forces, 0);

		  vx_vec[i] = CppAD::Value(tire_tangent_vel / 0.098);
		  fx_vec[i] = CppAD::Value(forces[0]);
	  }
	  plt::plot(vx_vec, fx_vec, {{"color", "k"}});
	  plot_cross(-.001,.001, -.1,.1);
	  plt::xlabel("Tire Angular Velocity (m/s)");
	  plt::ylabel("Longitudinal Force (N)");
	  plt::show();
	  
  }


  TEST(TireNetwork, vy_fy_plot)
  {
	  TireNetwork tire_network;
	  tire_network.load_model();
	  
	  VectorAD params = VectorAD::Zero(tire_network.getNumParams());
	  if(!loadVec(params, "/home/justin/tire.net"))
	  {
		  tire_network.setParams(params, 0);
	  }
	  
	  Eigen::Matrix<ADF,8,1> features;
	  Eigen::Matrix<ADF,TireNetwork::num_out_features,1> forces;
	  
	  features[0] = 0.2;
	  features[1] = 0;
	  features[2] = 0.1;
	  features[3] = 0.001;
    
	  features[4] = 29.76;
	  features[5] = 2083.0;
	  features[6] = 0.8;
	  features[7] = 0.0;
	  
	  int num = 11;
	  int len = 10000;
	  std::vector<float> vy_vec(len);
	  std::vector<float> fy_vec(len);
	  std::vector<float> tanh_vec(len);

	  for(int j = 0; j < num; j++)
	  {
		  ADF vx = (float) j/(num-1);
		  for(int i = 0; i < len; i++)
		  {
			  ADF vy = 1.0 * ADF((2.0*i/(float)len) - 1.0);
			  features[0] = vx;
			  features[1] = vy;

			  tire_network.forward(features, forces, 0);

			  tanh_vec[i] = CppAD::Value(40*CppAD::tanh(100*vy));
			  vy_vec[i] = CppAD::Value(vy);
			  fy_vec[i] = CppAD::Value(forces[1]);
		  }
		  
		  std::stringstream stream;
		  stream << std::fixed << std::setprecision(2) << CppAD::Value(vx);
		  std::string label = stream.str();
		  
		  float grey = (0.8*j) / num;
		  plt::plot(vy_vec, fy_vec, {{"color", std::to_string(grey)}, {"label", label}});
	  }
	  
	  plt::legend();
	  plt::xlabel("Lateral Velocity (m/s)");
	  plt::ylabel("Lateral Force (N)");
	  plt::show();
  }

  TEST(TireNetwork, vz_fz_plot)
  {
	  TireNetwork tire_network;
	  tire_network.load_model();
	  
	  VectorAD params = VectorAD::Zero(tire_network.getNumParams());
	  if(!loadVec(params, "/home/justin/tire.net"))
	  {
		  tire_network.setParams(params, 0);
	  }
	  
	  Eigen::Matrix<ADF,8,1> features;
	  Eigen::Matrix<ADF,TireNetwork::num_out_features,1> forces;
	  
	  features[0] = 0.0;
	  features[1] = 0;
	  features[2] = 0.0;
	  features[3] = 0.005;
    
	  features[4] = 29.76;
	  features[5] = 2083.0;
	  features[6] = 0.8;
	  features[7] = 0.0;
	  
	  int num = 1;
	  int len = 1000;
	  std::vector<float> sinkage_vec(len);
	  std::vector<float> fz_vec(len);
	  
	  for(int i = 0; i < len; i++)
	  {
		  ADF zr = 0.01 * ADF((2.0*i/(float)len) - 0.5);
		  features[3] = zr;
		  
		  tire_network.forward(features, forces, 0);
		  
		  sinkage_vec[i] = CppAD::Value(zr);
		  fz_vec[i] = CppAD::Value(forces[2]);
	  }
	  
	  plt::plot(sinkage_vec, fz_vec, {{"color", "k"}});
	  plt::xlabel("Tire Contact Height Error (m)");
	  plt::ylabel("Normal Force (N)");
	  plt::show();
  }
}

