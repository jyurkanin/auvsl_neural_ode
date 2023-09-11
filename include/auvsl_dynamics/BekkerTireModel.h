#pragma once

#include <vector>
#include <cmath>
#include <Eigen/Dense>

#include "generated/forward_dynamics.h"
#include "generated/model_constants.h"
#include "types/Scalars.h"


class BekkerTireModel
{
private:
	ADF k0;
	ADF Au;
	ADF K;
	ADF density;
	ADF c;
	ADF pg;
  
	ADF zr;
	ADF slip_ratio;
	ADF slip_angle;
  
	ADF kc;
	ADF kphi;
	ADF n0;
	ADF n1;
	ADF phi;

	ADF n;
	ADF pgc;
	const ADF R{0.098};
	const ADF b{0.05};
	ADF ze;
	ADF zu;
	ADF ku;
  
	ADF theta_f;
	ADF theta_r;
	ADF theta_c;
  
	const int num_steps = 100;
	
	ADF sigma_x_cf(ADF theta);
	ADF sigma_x_cc(ADF theta);
	ADF sigma_x_cr(ADF theta);  
	ADF jx_cf(ADF theta);
	ADF jx_cc(ADF theta);
	ADF jx_cr(ADF theta);
  
	ADF jy_cf(ADF theta);
	ADF jy_cc(ADF theta);
	ADF jy_cr(ADF theta);
  
	ADF j_cf(ADF theta);
	ADF j_cc(ADF theta);
	ADF j_cr(ADF theta);
  
	ADF tau_x_cf(ADF theta);
	ADF tau_x_cc(ADF theta);
	ADF tau_x_cr(ADF theta);  
	ADF tau_y_cf(ADF theta);
	ADF tau_y_cc(ADF theta);
	ADF tau_y_cr(ADF theta);

	ADF Fy_eqn3(ADF theta);
  
	ADF Fz_eqn1(ADF theta);
	ADF Fz_eqn2(ADF theta);
  
	ADF Fx_eqn1(ADF theta);
	ADF Fx_eqn2(ADF theta);
	ADF Fx_eqn3(ADF theta);
  
	ADF Ty_eqn1(ADF theta);

	ADF integrate(ADF (BekkerTireModel::*func)(ADF), ADF upper_b, ADF lower_b);
	
public:
	BekkerTireModel();
	Eigen::Matrix<ADF,4,1> get_forces(const Eigen::Matrix<ADF,8,1> &in_vec);
};




