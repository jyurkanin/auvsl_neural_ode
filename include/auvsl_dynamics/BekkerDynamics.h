#pragma once
#include "HybridDynamics.h"
#include "BekkerTireModel.h"

class BekkerDynamics : HybridDynamics
{
public:
	BekkerDynamics();
	~BekkerDynamics();

	void get_tire_f_ext(const Eigen::Matrix<Scalar,STATE_DIM,1> &X, LinkDataMap<Force> &ext_forces);

	void setParams(const VectorS &params);
	void getParams(VectorS &params);
private:
	Eigen::Matrix<Scalar,5,1> m_params;
	BekkerTireModel m_bekker_tire_model;
};
