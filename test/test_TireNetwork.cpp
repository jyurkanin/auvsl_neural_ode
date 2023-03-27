#include <matplotlibcpp.h>
#include <cpp_bptt.h>

#include "gtest/gtest.h"
#include "TireNetwork.h"
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
  
  TEST(TireNetwork, vx_fx_plot)
  {
    TireNetwork tire_network;
    Eigen::Matrix<ADF,TireNetwork::num_in_features,1> features;
    Eigen::Matrix<ADF,TireNetwork::num_out_features,1> forces;
    
    features[0] = 0;
    features[1] = 0;
    features[2] = 0;
    features[3] = 1;
    features[4] = 0.05;

    ADF tire_tangent_vel = features[3] * Jackal::rcg::tire_radius;
    
    int len = 1000;
    std::vector<float> vx_vec(len);
    std::vector<float> fx_vec(len);
    
    for(int i = 0; i < len; i++)
    {
      ADF vx = .01*ADF((2.0*i/(float)len) - 1.0) + tire_tangent_vel;
      features[0] = CppAD::abs(vx);
      
      tire_network.forward(features, forces);
      
      forces[0] = CppAD::abs(forces[0])*CppAD::tanh((tire_tangent_vel - vx));
      
      vx_vec[i] = CppAD::Value(vx - tire_tangent_vel);
      fx_vec[i] = CppAD::Value(forces[0]);
    }

    plot_cross(-1,1, -100,100);
    plt::plot(vx_vec, fx_vec, "r");
    plt::title("Vx vs Fx");
    plt::show();
  }
  
}

