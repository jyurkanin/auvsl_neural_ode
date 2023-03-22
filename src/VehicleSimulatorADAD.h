#pragma once

#include <cpp_bptt.h>
#include <types/Tensors.h>
#include <types/Scalars.h>
#include <vector>
#include "VehicleSystem.h"


// This class' only purpose is to overide the integrate function
class VehicleSimulatorADAD : public cpp_bptt::SimulatorADAD
{
public:
  VehicleSimulatorADAD(std::shared_ptr<VehicleSystem<ADAD>> sys) : SimulatorADAD(sys), m_system(sys)
  {
    
  }
  
  ~VehicleSimulatorADAD()
  {
    
  }

  // This overrides the Simulator base class function.
  virtual void integrate(const VectorADAD &Xk, VectorADAD &Xk1)
  {
    m_system->integrate(Xk, Xk1);
  }
  
private:
  std::shared_ptr<VehicleSystem<ADAD>> m_system;
};
