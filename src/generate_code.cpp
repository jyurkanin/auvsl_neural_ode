#include <memory>
#include <fenv.h>
#include <iostream>

#include <cpp_bptt.h>

#include "VehicleSystem.h"
#include "VehicleSimulatorADAD.h"


int main()
{
  std::shared_ptr<VehicleSystem<ADAD>> vehicle_system = std::make_shared<VehicleSystem<ADAD>>();
  std::shared_ptr<VehicleSimulatorADAD> simulator = std::make_shared<VehicleSimulatorADAD>(vehicle_system);
  
  cpp_bptt::Generator generator;
  generator.setSimulator(simulator);
  generator.initialize();
}
