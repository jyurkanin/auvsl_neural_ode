#include <matplotlibcpp.h>
#include <cpp_bptt.h>

#include "gtest/gtest.h"
#include "TireNetwork.h"
#include "VehicleSystem.h"
#include "generated/model_constants.h"

namespace plt = matplotlibcpp;

namespace{
  void plot_cross(float x1, float x2, float y1, float y2)
  {
	  std::vector<float> x(2);
	  std::vector<float> y(2);

	  x[0] = x1;
	  x[1] = x2;
	  y[0] = 0;
	  y[1] = 0;
	  plt::plot(x, y, "b");
	  x[0] = 0;
	  x[1] = 0;
	  y[0] = y1;
	  y[1] = y2;
	  plt::plot(x, y, "b");    
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
    VehicleSystem<ADF> system1;
    VehicleSystem<ADF> system2;
    
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
	  loadVec(params, "/home/justin/tire.net");
	  tire_network.setParams(params, 0);
	  
	  Eigen::Matrix<ADF,8,1> features;
	  Eigen::Matrix<ADF,TireNetwork::num_out_features,1> forces;
	  
	  features[0] = 0;
	  features[1] = 0;
	  features[2] = 0;
	  features[3] = 0.005;
    
	  features[4] = 29.758547;
	  features[5] = 2083.0;
	  features[6] = 1.197933;
	  features[7] = 0.102483;
    
	  int len = 10000;
	  std::vector<float> vx_vec(len);
	  std::vector<float> fx_vec(len);

	  for(int j = 0; j < 10; j++)
	  {
		  ADF tire_tangent_vel = j/4.0;
		  for(int i = 0; i < len; i++)
		  {
			  ADF tire_vx = 1.0*ADF((2.0*i/(float)len) - 1.0) + tire_tangent_vel;

			  features[0] = tire_vx;
			  features[2] = tire_tangent_vel/.098;
	
			  tire_network.forward(features, forces, 0);
	
			  vx_vec[i] = CppAD::Value(tire_tangent_vel - tire_vx);
			  fx_vec[i] = CppAD::Value(forces[0]);
		  }
      
		  plot_cross(-1,1, -100,100);
		  plt::plot(vx_vec, fx_vec);
	  }

	  plt::title("Diff vs Fx");
	  plt::show();
  }


  TEST(TireNetwork, vy_fy_plot)
  {
	  TireNetwork tire_network;
	  tire_network.load_model();
	  
	  VectorAD params = VectorAD::Zero(tire_network.getNumParams());
	  loadVec(params, "/home/justin/tire.net");
	  tire_network.setParams(params, 0);
	  
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
    
	  int len = 10000;
	  std::vector<float> vy_vec(len);
	  std::vector<float> fy_vec(len);
	  std::vector<float> tanh_vec(len);

	  for(int j = 0; j < 8; j++)
	  {
		  ADF vx = j/8.0;
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
      
		  plot_cross(-1,1, -10,10);
		  //plt::plot(vy_vec, tanh_vec, "g");
		  plt::plot(vy_vec, fy_vec);
	  }
	  plt::title("Vy vs Fy");
	  plt::show();
  }

  TEST(TireNetwork, vz_fz_plot)
  {
	  TireNetwork tire_network;
	  tire_network.load_model();
	  
	  VectorAD params = VectorAD::Zero(tire_network.getNumParams());
	  loadVec(params, "/home/justin/tire.net");
	  tire_network.setParams(params, 0);
	  
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
    
	  int len = 10000;
	  std::vector<float> sinkage_vec(len);
	  std::vector<float> fz_vec(len);

	  for(int j = 0; j < 8; j++)
	  {
		  features[0] = 0; //j/8.0;
		  for(int i = 0; i < len; i++)
		  {
			  ADF zr = 0.1 * ADF((2.0*i/(float)len) - 1.0);
			  features[3] = zr;
	
			  tire_network.forward(features, forces, 0);
	
			  sinkage_vec[i] = CppAD::Value(zr);
			  fz_vec[i] = CppAD::Value(forces[2]);
		  }
      
		  //plot_cross(-1,1, -10,10);
		  plt::plot(sinkage_vec, fz_vec);
	  }
    
	  plt::title("Zr vs Fz");
	  plt::show();
  }
}

