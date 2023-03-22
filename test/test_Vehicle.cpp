#include <vector>
#include "gtest/gtest.h"

#include <cpp_bptt.h>
#include <cpp_neural.h>

#include "VehicleSystem.h"
#include "VehicleSimulatorF.h"
#include "VehicleSimulatorAD.h"



namespace {
  // using cpp_bptt::VectorF;
  // using cpp_bptt::MatrixF;
  // using cpp_bptt::VectorAD;
  // using cpp_bptt::MatrixAD;
  // using cpp_bptt::ADF;
  // using cpp_bptt::ADAD;




  class VehicleFixture : public ::testing::Test
  {
  public:
    std::vector<VectorF> m_gt_list;
    std::vector<VectorAD> m_gt_list_adf;
    VectorF m_x0;
    VectorF m_params;
    
    std::shared_ptr<VehicleSystem<ADF>>    m_system_adf;
    std::shared_ptr<VehicleSimulatorAD>    m_simulator_adf;
    std::shared_ptr<VehicleSystem<float>>  m_system_f;
    std::shared_ptr<VehicleSimulatorF>     m_simulator_f;
    
    VehicleFixture()
    {
      srand(time(NULL)); // randomize seed
      
      m_system_adf = std::make_shared<VehicleSystem<ADF>>();
      m_simulator_adf = std::make_shared<VehicleSimulatorAD>(m_system_adf);
      
      m_system_f = std::make_shared<VehicleSystem<float>>();
      m_simulator_f = std::make_shared<VehicleSimulatorF>(m_system_f);

      m_params = VectorF::Random(m_system_adf->getNumParams());
      m_x0 = VectorF::Random(m_system_adf->getStateDim());
      m_gt_list.resize(m_system_adf->getNumSteps());
      m_gt_list_adf.resize(m_system_adf->getNumSteps());

      for(int i = 0; i < m_gt_list.size(); i++)
      {
	m_gt_list[i] = VectorF::Random(m_system_f->getStateDim());
	m_gt_list_adf[i] = VectorAD::Zero(m_system_f->getStateDim());
	for(int j = 0; j < m_x0.size(); j++)
	{
	  m_gt_list_adf[i][j] = m_gt_list[i][j]; 
	}
      }
      
    }
    ~VehicleFixture(){}
        
    VectorF getGradientHard()
    {
      VectorF x0(m_system_f->getStateDim());
      for(int i = 0; i < x0.size(); i++)
      {
	x0[i] = m_x0[i];
      }
      
      VectorF params(m_system_f->getNumParams());
      for(int i = 0; i < params.size(); i++)
      {
	params[i] = m_params[i];
      }
            
      m_system_f->setParams(params);
      
      VectorF gradient;
      float loss;
      m_simulator_f->forward_backward(x0, m_gt_list, gradient, loss);
      
      return gradient;
    }
    
  };

  
  TEST_F(VehicleFixture, validate_gradient_hard)
  {
    VectorF grad_hard = getGradientHard();

    for(int i = 0; i < m_system_f->getNumParams(); i++)
    {
      std::cout << grad_hard[i] << "\n";
    }
  }
}
