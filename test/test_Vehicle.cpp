#include <vector>
#include "gtest/gtest.h"

#include <matplotlibcpp.h>
#include <cpp_bptt.h>
#include <cpp_neural.h>

#include "HybridDynamics.h"
#include "VehicleSystem.h"

namespace plt = matplotlibcpp;

namespace {

  class VehicleFixture : public ::testing::Test
  {
  public:
    std::vector<VectorF> m_gt_list;
    std::vector<VectorAD> m_gt_list_adf;
    VectorAD m_x0;
    VectorAD m_params;

    std::shared_ptr<VehicleSystem<ADF>>    m_system_adf;
    
    VehicleFixture()
    {
      srand(time(NULL)); // randomize seed
      
      m_system_adf    = std::make_shared<VehicleSystem<ADF>>();
      
      m_params = VectorAD::Zero(m_system_adf->getNumParams());
      m_x0 = VectorAD::Zero(m_system_adf->getStateDim());
      
      m_system_adf->getDefaultParams(m_params);
      m_system_adf->getDefaultInitialState(m_x0);

      m_gt_list.resize(m_system_adf->getNumSteps());
      m_gt_list_adf.resize(m_system_adf->getNumSteps());
      
      for(int i = 0; i < m_gt_list.size(); i++)
      {
	m_gt_list[i] = VectorF::Zero(m_system_adf->getStateDim());
	m_gt_list_adf[i] = VectorAD::Random(m_system_adf->getStateDim());
	for(int j = 0; j < m_gt_list_adf[i].size(); j++)
	{
	  m_gt_list[i][j] = CppAD::Value(m_gt_list_adf[i][j]);
	}
      }
    }
    ~VehicleFixture(){}
    
    VectorF getGradientSimple()
    {      
      CppAD::Independent(m_params);
      m_system_adf->setParams(m_params);
      
      std::vector<VectorAD> x_list(m_system_adf->getNumSteps());
      
      VectorAD loss(1);
      
      loss[0] = 0;
      for(int i = 0; i < x_list.size(); i++)
      {
	loss[0] += m_system_adf->loss(m_gt_list_adf[i], x_list[i]);
      }
      
      CppAD::ADFun<double> func(m_params, loss);
      
      VectorF y0(1);
      y0[0] = 1;

      std::cout << "Loss " << CppAD::Value(loss[0]) << "\n";
      
      return func.Reverse(1, y0);
    }
  };

  


  
  TEST_F(VehicleFixture, validate_gradient_easy)
  {
    VectorF grad_simple = getGradientSimple();
    //VectorF grad_bptt = getGradientBPTT();

    for(int i = 0; i < m_system_adf->getNumParams(); i++)
    {
      
    }
  }
  
  
  TEST_F(VehicleFixture, settling)
  {
    int num_steps = 100;
    std::vector<double> time(num_steps);
    std::vector<double> elev(num_steps);
    std::vector<double> x_vec(num_steps);
    std::vector<double> y_vec(num_steps);
    
    m_system_adf->setParams(m_params);

    VectorAD xk(m_system_adf->getStateDim());
    VectorAD xk1(m_system_adf->getStateDim());
    
    xk = m_x0;
    
    for(int i = 0; i < num_steps; i++)
    {
      xk[HybridDynamics::STATE_DIM] = 0.0; //vl
      xk[HybridDynamics::STATE_DIM+1] = 0.0; //vr
      m_system_adf->integrate(xk, xk1);
      
      time[i] = i * CppAD::Value(HybridDynamics::timestep);
      elev[i] = CppAD::Value(xk1[6]);
      x_vec[i] = CppAD::Value(xk1[4]);
      y_vec[i] = CppAD::Value(xk1[5]);
      xk = xk1;
    }
    
    plt::subplot(1,2,1);
    plt::title("X-Y plot");
    plt::plot(x_vec, y_vec);
    
    plt::subplot(1,2,2);
    plt::title("Time vs Elevation");
    plt::plot(time, elev);
    
    plt::show();
  }
  
  
  TEST_F(VehicleFixture, straight)
  {
    int num_steps = 100;
    std::vector<double> time(num_steps);
    std::vector<double> elev(num_steps);
    std::vector<double> x_vec(num_steps);
    std::vector<double> y_vec(num_steps);
    
    m_system_adf->setParams(m_params);

    VectorAD xk(m_system_adf->getStateDim());
    VectorAD xk1(m_system_adf->getStateDim());
    
    xk = m_x0;
    
    for(int i = 0; i < num_steps; i++)
    {
      xk[HybridDynamics::STATE_DIM] = 1; //vl
      xk[HybridDynamics::STATE_DIM+1] = 1; //vr
      m_system_adf->integrate(xk, xk1);
      
      time[i] = i * 0.1;
      elev[i] = CppAD::Value(xk1[6]);
      x_vec[i] = CppAD::Value(xk1[4]);
      y_vec[i] = CppAD::Value(xk1[5]);
      xk = xk1;
    }
    
    plt::subplot(1,2,1);
    plt::title("X-Y plot");
    plt::xlabel("[m]");
    plt::ylabel("[m]");
    plt::plot(x_vec, y_vec);
    
    plt::subplot(1,2,2);
    plt::title("Time vs Elevation");
    plt::xlabel("Time [s]");
    plt::ylabel("Elevation [m]");
    plt::plot(time, elev);
    
    plt::show();

    // tire_radius is .098m * 1rad/s
    std::cout << "Distance: " << x_vec.back() << "\n";
    EXPECT_NEAR(x_vec.back(), 10*.098, 5e-2);
  }

  TEST_F(VehicleFixture, circle)
  {
    int num_steps = 1000;
    std::vector<double> time(num_steps);
    std::vector<double> elev(num_steps);
    std::vector<double> x_vec(num_steps);
    std::vector<double> y_vec(num_steps);
    
    m_system_adf->setParams(m_params);

    VectorAD xk(m_system_adf->getStateDim());
    VectorAD xk1(m_system_adf->getStateDim());
    
    xk = m_x0;

    double max_x = CppAD::Value(xk[4]);
    double min_x = CppAD::Value(xk[4]);
    double max_y = CppAD::Value(xk[5]);
    double min_y = CppAD::Value(xk[5]);
    
    for(int i = 0; i < num_steps; i++)
    {
      xk[HybridDynamics::STATE_DIM] = 2; //vl
      xk[HybridDynamics::STATE_DIM+1] = 1; //vr
      m_system_adf->integrate(xk, xk1);
      
      time[i] = i * 0.1;
      elev[i] = CppAD::Value(xk1[6]);
      x_vec[i] = CppAD::Value(xk1[4]);
      y_vec[i] = CppAD::Value(xk1[5]);
      xk = xk1;

      if(x_vec[i] > max_x)
      {
	max_x = x_vec[i];
      }
      if(x_vec[i] < min_x)
      {
	min_x = x_vec[i];
      }
      
      if(y_vec[i] > max_y)
      {
	max_y = y_vec[i];
      }
      if(y_vec[i] < min_y)
      {
	min_y = y_vec[i];
      }
    }
    
    plt::subplot(1,2,1);
    plt::title("X-Y plot");
    plt::xlabel("[m]");
    plt::ylabel("[m]");
    plt::plot(x_vec, y_vec);
    
    plt::subplot(1,2,2);
    plt::title("Time vs Elevation");
    plt::xlabel("Time [s]");
    plt::ylabel("Elevation [m]");
    plt::plot(time, elev);
    
    plt::show();

    EXPECT_NEAR((max_x - min_x), (max_y - min_y), 1e-2);
  }

  TEST_F(VehicleFixture, explosion)
  {
    VectorAD gt_vec = VectorAD::Zero(m_system_adf->getStateDim());
    gt_vec[4] = 7.92647e-05;
    gt_vec[5] = -6.47106e-05;
    CppAD::Independent(m_params);
    m_system_adf->setParams(m_params);
    
    double xk[] = {-0.00302435, 0.0473666, 0.00272453, 0.998869, 0.000168885, -0.000122925, -0.0547984, 0, 0, 0, 0, 0, 0, -0.0513761, -0.000988309, -0.000343537, 0, 0, 0, 0, 0, 0, 0};
    
    Eigen::Matrix<ADF, HybridDynamics::STATE_DIM, 1> model_x0;
    for(int i = 0; i < HybridDynamics::STATE_DIM; i++)
    {
      model_x0[i] = xk[i];
    }
    Eigen::Matrix<ADF, HybridDynamics::CNTRL_DIM, 1> model_u;
    model_u[0] = 0.0;
    model_u[1] = 0.0;
    Eigen::Matrix<ADF, HybridDynamics::STATE_DIM, 1> model_x1;
    
    const int num_steps = 100; // 100*.001 = .1
    for(int ii = 0; ii < num_steps; ii++)
    {
      model_x0[17] = model_x0[19] = model_u[0];
      model_x0[18] = model_x0[20] = model_u[1];
      
      m_system_adf->m_hybrid_dynamics.RK4(model_x0, model_x1, model_u);
      model_x0 = model_x1;
    }

    VectorAD loss_ad(1);
    ADF x_err = gt_vec[4] - model_x1[4];
    ADF y_err = gt_vec[5] - model_x1[5];
    loss_ad[0] = (x_err*x_err) + (y_err*y_err);
    CppAD::ADFun<double> func(m_params, loss_ad);

    VectorF y0(1);
    y0[0] = 1;
    VectorF gradient = func.Reverse(1, y0);

    for(int i = 0; i < gradient.size(); i++)
    {
      std::cout << "d_param " << gradient[i] << "\n";
    }
    
    
  }
}
