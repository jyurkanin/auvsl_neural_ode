#include "VehicleSimulatorF.h"




VehicleSimulatorF::VehicleSimulatorF(std::shared_ptr<cpp_bptt::System<float>> sys) : cpp_bptt::SimulatorF(sys) {}
VehicleSimulatorF::~VehicleSimulatorF() {}
 
#include "partials.cpp"
