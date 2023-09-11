#pragma once

#include <Eigen/Dense>
#include "types/Scalars.h"
#include "types/GroundTruthDataRow.h"

template<typename Scalar>
class System
{
public:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;

	int getControlDim() {return m_num_controls;}
	int getStateDim() {return m_num_states;}
	int getNumParams() {return m_num_params;}
	
	virtual void   getDefaultParams(VectorS &params) = 0;
	virtual void   getDefaultInitialState(VectorS &state) = 0;
	virtual void   setParams(const VectorS &params) = 0;
	virtual void   getParams(VectorS &params) = 0;
	virtual void   forward(const VectorS &X, VectorS &Xd) = 0;
	virtual Scalar loss(const VectorS &gt_vec, VectorS &vec) = 0;

	virtual void evaluate(const VectorS &gt_vec, const VectorS &vec, Scalar &ang_err, Scalar &lin_err) = 0;
	virtual void integrate(const VectorS &X0, VectorS &X1) = 0;
	virtual VectorS initializeState(const GroundTruthDataRow &gt_state) = 0;
protected:
	void setControlDim(int num_controls) {m_num_controls = num_controls;}
	void setStateDim(int num_states) {m_num_states = num_states;}
	void setNumParams(int num_params) {m_num_params = num_params;}
	
private:
	int m_num_controls;
	int m_num_states;
	int m_num_params;
};
