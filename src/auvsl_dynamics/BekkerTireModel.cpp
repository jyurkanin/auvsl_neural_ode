#include "BekkerTireModel.h"



BekkerTireModel::BekkerTireModel()
{
	k0 = 2195;
	Au = 192400;
	K = .0254;
	density = 1681;
	c = 8.62;
	pg = 100; //15psi
}



ADF BekkerTireModel::sigma_x_cf(ADF theta)
{
	//in some cases, CppAD::cos - CppAD::cos can return negative due to theta != theta_f, but very close.
	ADF diff = CppAD::cos(theta) - CppAD::cos(theta_f); //std::max(0.0f,CppAD::cos(theta) - CppAD::cos(theta_f));
	// diff = CppAD::CondExpLt(diff, ADF(0.0), ADF(0.0), diff);
	ADF temp = ((kc/b) + kphi)*CppAD::pow(R*diff, n);
	return temp;
}

ADF BekkerTireModel::sigma_x_cc(ADF theta)
{
	return ((kc/b)+kphi)*CppAD::pow(ze, n);
}

ADF BekkerTireModel::sigma_x_cr(ADF theta){
	ADF diff = CppAD::cos(theta) - CppAD::cos(theta_r);
	ADF temp = ku*CppAD::pow(R*(diff), n);
	return temp;
}
  
ADF BekkerTireModel::jx_cf(ADF theta){
	ADF temp = R*((theta - theta_f) - (1 - slip_ratio)*(CppAD::sin(theta)-CppAD::sin(theta_f)));
	return temp;
}

ADF BekkerTireModel::jx_cc(ADF theta)
{
	return R*((theta_c - theta_f) + (1 - slip_ratio)*CppAD::sin(theta_f) - CppAD::sin(theta_c) + (slip_ratio*CppAD::cos(theta_c)*CppAD::tan(theta)));
}
ADF BekkerTireModel::jx_cr(ADF theta)
{
	return R*((2*theta_c + theta - theta_f) - (1 - slip_ratio)*(CppAD::sin(theta) - CppAD::sin(theta_f)) - 2*CppAD::sin(theta_c));
}
  
ADF BekkerTireModel::jy_cf(ADF theta)
{
	return R*(1-slip_ratio) * CppAD::tan(slip_angle) * (theta_f - theta);
}
ADF BekkerTireModel::jy_cc(ADF theta)
{
	return R*(1-slip_ratio) * CppAD::tan(slip_angle) * (theta_f - theta + CppAD::sin(theta_c) - CppAD::cos(theta_c)*CppAD::tan(theta));
}
ADF BekkerTireModel::jy_cr(ADF theta)
{
	return R*(1-slip_ratio) * CppAD::tan(slip_angle) * (theta_f - theta - 2*theta_c + 2*CppAD::sin(theta_c));
}
  
ADF BekkerTireModel::j_cf(ADF theta)
{
	ADF temp = CppAD::sqrt(CppAD::pow(jy_cf(theta), 2) + CppAD::pow(jx_cf(theta), 2));
	return temp;
}

ADF BekkerTireModel::j_cc(ADF theta)
{
	return CppAD::sqrt(CppAD::pow(jy_cc(theta), 2) + CppAD::pow(jx_cc(theta), 2));
}

ADF BekkerTireModel::j_cr(ADF theta)
{
	return CppAD::sqrt(CppAD::pow(jy_cr(theta), 2) + CppAD::pow(jx_cr(theta), 2));
}
  
ADF BekkerTireModel::tau_x_cf(ADF theta)
{
	//smart divide
	ADF j_cf_theta = j_cf(theta);
	j_cf_theta = CppAD::CondExpEq(j_cf_theta, ADF(0.0), ADF(1e-6), j_cf_theta);
	return (-jx_cf(theta)/j_cf_theta) * (c + (sigma_x_cf(theta)*CppAD::tan(phi))) * (1 - (exp(-j_cf_theta/K)));
}

ADF BekkerTireModel::tau_x_cc(ADF theta)
{
	ADF j_cc_theta = j_cc(theta);
	j_cc_theta = CppAD::CondExpEq(j_cc_theta, ADF(0.0), ADF(1e-6), j_cc_theta);
	
	ADF temp = jx_cc(theta)/j_cc_theta;
	return (-temp) * (c + (sigma_x_cc(theta)*CppAD::tan(phi))) * (1 - (exp(-j_cc_theta/K)));
}

ADF BekkerTireModel::tau_x_cr(ADF theta)
{
	ADF j_cr_theta = j_cr(theta);
	j_cr_theta = CppAD::CondExpEq(j_cr_theta, ADF(0.0), ADF(1e-6), j_cr_theta);
	ADF temp = jx_cr(theta)/j_cr_theta;
	
	return (-temp) * (c + (sigma_x_cr(theta)*CppAD::tan(phi))) * (1 - (exp(-j_cr_theta/K)));
}

ADF BekkerTireModel::tau_y_cf(ADF theta)
{
	ADF j_cf_theta = j_cf(theta);
	j_cf_theta = CppAD::CondExpEq(j_cf_theta, ADF(0.0), ADF(1e-6), j_cf_theta);
	ADF temp = jy_cf(theta)/j_cf_theta;
	
	return (-temp) * (c + (sigma_x_cf(theta)*CppAD::tan(phi))) * (1 - (exp(-j_cf_theta/K)));
}
ADF BekkerTireModel::tau_y_cc(ADF theta)
{
	ADF j_cc_theta = j_cc(theta);
	j_cc_theta = CppAD::CondExpEq(j_cc_theta, ADF(0.0), ADF(1e-6), j_cc_theta);
	ADF temp = jy_cc(theta)/j_cc_theta;
	
	return (-temp) * (c + (sigma_x_cc(theta)*CppAD::tan(phi))) * (1 - (exp(-j_cc_theta/K)));
}
ADF BekkerTireModel::tau_y_cr(ADF theta)
{
	ADF j_cr_theta = j_cr(theta);
	j_cr_theta = CppAD::CondExpEq(j_cr_theta, ADF(0.0), ADF(1e-6), j_cr_theta);
	ADF temp = jy_cr(theta)/j_cr_theta;
	
	return (-temp) * (c + (sigma_x_cr(theta)*CppAD::tan(phi))) * (1 - (exp(-j_cr_theta/K)));
}


ADF BekkerTireModel::Fy_eqn3(ADF theta)
{
	return tau_y_cc(theta)*CppAD::pow(1.0f/CppAD::cos(theta), 2);
}
  
ADF BekkerTireModel::Fz_eqn1(ADF theta)
{
	return (sigma_x_cf(theta)*CppAD::cos(theta)) + (tau_x_cf(theta)*CppAD::sin(theta));
}
ADF BekkerTireModel::Fz_eqn2(ADF theta)
{
	return (sigma_x_cr(theta)*CppAD::cos(theta)) + (tau_x_cr(theta)*CppAD::sin(theta));
}
  
ADF BekkerTireModel::Fx_eqn1(ADF theta)
{
	return (tau_x_cf(theta)*CppAD::cos(theta)) - (sigma_x_cf(theta)*CppAD::sin(theta));
}

ADF BekkerTireModel::Fx_eqn2(ADF theta)
{
	return (tau_x_cr(theta)*CppAD::cos(theta)) - (sigma_x_cr(theta)*CppAD::sin(theta));
}
ADF BekkerTireModel::Fx_eqn3(ADF theta)
{
	return tau_x_cc(theta)*(CppAD::pow(1.0f/CppAD::cos(theta), 2));
} 
ADF BekkerTireModel::Ty_eqn1(ADF theta)
{
	return tau_x_cc(theta)*(CppAD::pow(1.0f/CppAD::cos(theta), 2));
}
ADF BekkerTireModel::integrate(ADF (BekkerTireModel::*func)(ADF), ADF upper_b, ADF lower_b)
{
	ADF dtheta = (upper_b - lower_b) / num_steps;
	ADF eps = dtheta*.1; //adaptive machine epsilon
    
	//trapezoidal rule.
	ADF sum = 0;
	for(ADF theta = lower_b; theta < (upper_b - eps - dtheta); theta += dtheta){
		sum += .5*dtheta*((this->*func)(theta + dtheta) + (this->*func)(theta));
	}

	//last iteration is different to ensure no ADFing point error occurs.
	//This ensures integration goes exactly to the upper bound and does not exceed it in the slightest.
	sum += .5*dtheta*((this->*func)(upper_b) + (this->*func)(upper_b - dtheta));
    
	return sum;
}



Eigen::Matrix<ADF,4,1> BekkerTireModel::get_forces(const Eigen::Matrix<ADF,8,1> &features)
{
	ADF Fx = 0;
	ADF Fy = 0;
	ADF Fz = 0;
	ADF Ty = 0;

	zr = CppAD::CondExpGt(features[0], R, R, features[0]);
	slip_ratio = features[1];
	slip_angle = features[2];
  
	kc = features[3];
	kphi = features[4];
	n0 = features[5];
	n1 = features[6];
	phi = features[7];
  
	n = n0 + (n1 * CppAD::abs(slip_ratio));
	pgc = ((kc/b) + kphi) * CppAD::pow(zr, n);
	
	if(pgc < pg)
	{
		ze = zr;
	}
	else
	{
		ze = CppAD::pow(pg/((kc/b) + kphi), 1.0f/n);
	}
  
	ku = k0 + (Au*ze);
	zu = CppAD::pow(((kc/b) + kphi) / ku, 1.0f/n) * ze;
  
	theta_f =  CppAD::acos(1 - (zr/R));
	theta_r = -CppAD::acos(1 - ((zu + zr - ze)/R));
	theta_c =  CppAD::acos(1 - ((zr - ze)/R));
    
	ADF bt_R = R * b;
	ADF bt_RR = R * R * b;
  
	Fx = bt_R*integrate(&BekkerTireModel::Fx_eqn1, theta_f, theta_c) +
		bt_R*CppAD::cos(theta_c)*integrate(&BekkerTireModel::Fx_eqn3, theta_c, -theta_c) +
		bt_R*integrate(&BekkerTireModel::Fx_eqn2, -theta_c, theta_r);
  
	Fy = bt_R*integrate(&BekkerTireModel::tau_y_cf, theta_f, theta_c) +
		bt_R*CppAD::cos(theta_c)*integrate(&BekkerTireModel::Fy_eqn3, theta_c, -theta_c) + //original matlab code has a bug here. In the matlab code I integrated from -theta_c to theta_c which gives a negative result.
		bt_R*integrate(&BekkerTireModel::tau_y_cr, -theta_c, theta_r);
  
	Fz = bt_R*integrate(&BekkerTireModel::Fz_eqn1, theta_f, theta_c) +
		bt_R*integrate(&BekkerTireModel::Fz_eqn2, -theta_c, theta_r) +
		2*bt_R*CppAD::sin(theta_c)*pg;
  
	Ty = -bt_RR*integrate(&BekkerTireModel::tau_x_cf, theta_f, theta_c) + 
		-bt_RR*integrate(&BekkerTireModel::tau_x_cr, -theta_c, theta_r) +
		-bt_RR*CppAD::cos(theta_c)*CppAD::cos(theta_c)*integrate(&BekkerTireModel::Ty_eqn1, theta_c, -theta_c);

	Ty = 1000*Ty;
	Fx = 1000*Fx;
	Fy = 1000*Fy;
	Fz = 1000*Fz;
	Eigen::Matrix<ADF,4,1> tire_wrench;
	tire_wrench[0] = Fx;
	tire_wrench[1] = Fy;
	tire_wrench[2] = Fz;
	tire_wrench[3] = 0.0;
  
	return tire_wrench;
}
