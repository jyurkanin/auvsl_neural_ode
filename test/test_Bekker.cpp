#include <matplotlibcpp.h>

#include "gtest/gtest.h"
#include "BekkerTireModel.h"
#include "generated/model_constants.h"
#include "types/Scalars.h"

#include <iostream>

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
  
  TEST(TireNetwork, vx_fx_plot)
  {
	  BekkerTireModel tire_model;
	  
	  Eigen::Matrix<ADF,8,1> features;
	  Eigen::Matrix<ADF,4,1> forces;
	  
	  features[0] = 0;
	  features[1] = 0;
	  features[2] = 0;
	  
	  features[3] = 29.758547;
	  features[4] = 2083.0;
	  features[5] = 1.197933;
	  features[6] = 0.102483;
	  features[7] = 0.652405;
    
	  int len = 10000;
	  std::vector<float> vx_vec(len);
	  std::vector<float> fx_vec(len);

	  for(int j = 0; j < 10; j++)
	  {
		  ADF tire_tangent_vel = j/4.0;
		  for(int i = 0; i < len; i++)
		  {
			  ADF tire_vx = 1.0*ADF((2.0*i/(float)len) - 1.0) + tire_tangent_vel;
			  ADF slip_ratio = (tire_tangent_vel - tire_vx) / CppAD::abs(tire_tangent_vel);

			  if(tire_tangent_vel == 0.0)
			  {
				  if(tire_vx == 0.0)
				  {
					  slip_ratio = 0.0;
				  }
				  else
				  {
					  slip_ratio = 1.0 - (tire_vx/1e-3);
				  }
			  }
			  else
			  {
				  if(tire_vx == 0.0)
				  {
					  slip_ratio = 1.0 - (1e-3/tire_tangent_vel);
				  }
				  else
				  {
					  slip_ratio = 1.0 - (tire_vx/tire_tangent_vel);
				  }				  
			  }
			  
			  features[0] = 0.005;
			  features[1] = slip_ratio;
			  features[2] = 0;
			  
			  forces = tire_model.get_forces(features);
			  
			  vx_vec[i] = CppAD::Value(features[1]);
			  fx_vec[i] = CppAD::Value(forces[0]);
		  }
      
		  plot_cross(-1,1, -100,100);
		  plt::plot(vx_vec, fx_vec);
	  }

	  plt::title("Slip vs Fx");
	  plt::show();
  }
}

