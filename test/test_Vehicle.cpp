#include <vector>
#include "gtest/gtest.h"

#include <cpp_bptt.h>
#include <cpp_neural.h>

#include "VehicleSystem.h"
#include "VehicleSimulatorF.h"
#include "VehicleSimulatorAD.h"



namespace {

  class VehicleFixture : public ::testing::Test
  {
  public:
    std::vector<VectorF> m_gt_list;
    std::vector<VectorAD> m_gt_list_adf;
    VectorAD m_x0;
    VectorAD m_params;

    std::shared_ptr<VehicleSystem<ADF>>    m_system_adf;
    std::shared_ptr<VehicleSimulatorAD>    m_simulator_adf;
    
    VehicleFixture()
    {
      srand(time(NULL)); // randomize seed
      
      m_system_adf    = std::make_shared<VehicleSystem<ADF>>();
      m_simulator_adf = std::make_shared<VehicleSimulatorAD>(m_system_adf);
      
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
      m_simulator_adf->forward(m_x0, x_list);
      
      VectorAD loss(1);
      
      loss[0] = 0;
      for(int i = 0; i < x_list.size(); i++)
      {
	loss[0] += m_system_adf->loss(m_gt_list_adf[i], x_list[i]);
      }
      
      CppAD::ADFun<float> func(m_params, loss);
      
      VectorF y0(1);
      y0[0] = 1;

      std::cout << "Loss " << loss[0] << "\n";
      
      return func.Reverse(1, y0);
    }
    
    VectorF getGradientBPTT()
    {
      m_system_adf->setParams(m_params);
      
      VectorF gradient;
      float loss;
      VectorF x0_f(m_system_adf->getStateDim());
      for(int i = 0; i < x0_f.size(); i++)
      {
	x0_f[i] = CppAD::Value(m_x0[i]);
      }
      
      m_simulator_adf->forward_backward(x0_f, m_gt_list, gradient, loss);
      
      std::cout << "Loss " << loss << "\n";
      
      return gradient;
    }    
  };

  


  
  TEST_F(VehicleFixture, validate_gradient_easy)
  {
    VectorF grad_simple = getGradientSimple();
    VectorF grad_bptt = getGradientBPTT();

    for(int i = 0; i < m_system_adf->getNumParams(); i++)
    {
      std::cout << grad_simple[i] << ", " << grad_bptt[i] << "\n";
      //EXPECT_LE(fabs(grad_simple[i] - grad_bptt[i]), fabs(1e-4f*grad_simple[i]));
    }
  }

}
