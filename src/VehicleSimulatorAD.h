#pragma once

#include <cpp_neural.h>

#include "VehicleSystem.h"
#include "SimulatorAD.h"

#include <vector>

// This class' only purpose is to overide the integrate function
class VehicleSimulatorAD : public cpp_bptt::SimulatorAD
{
public:
  VehicleSimulatorAD(std::shared_ptr<VehicleSystem<ADF>> sys) : cpp_bptt::SimulatorAD(sys), m_system(sys)
  {
    
  }
  ~VehicleSimulatorAD()
  {
    
  }

  // This overrides the Simulator base class function.
  virtual void integrate(const VectorAD &Xk, VectorAD &Xk1)
  {
    m_system->integrate(Xk, Xk1);
  }    
  
private:
  std::shared_ptr<VehicleSystem<ADF>> m_system;
};
