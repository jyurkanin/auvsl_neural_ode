#pragma once

#include <cpp_bptt.h>
#include <memory>

class VehicleSimulatorF : public cpp_bptt::SimulatorF
{
public:
  VehicleSimulatorF(std::shared_ptr<cpp_bptt::System<float>> sys);
  ~VehicleSimulatorF();

  // Implemented in SimulatorF
  // virtual void forward_backward(const VectorF &x0,
  // 				const std::vector<VectorF> &gt_list,
  // 				VectorF &gradient,
  // 				float &loss);

  
  void computePartials(const VectorF &xk0,
		       const VectorF &theta,
		       const VectorF &xk1_gt,
		       VectorF &xk1,
		       VectorF &partial_state_state,
		       VectorF &partial_state_param,
		       VectorF &partial_loss_params,
		       VectorF &partial_loss_state,
		       float   &loss);

};
