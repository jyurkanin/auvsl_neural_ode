#pragma once

#include "types/System.h"
#include "types/Scalars.h"

// I believe this is ready.
// This represents an ODE and loss function

///todo: implement the other base class functions
class LinearSystem : public System<ADF>
{
public:
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, 1> VectorAD;
	typedef Eigen::Matrix<ADF, Eigen::Dynamic, Eigen::Dynamic> MatrixAD;
	
	LinearSystem();
	~LinearSystem();

	virtual void getDefaultParams(VectorAD &params);
	virtual void getDefaultInitialState(VectorAD &state);
	virtual void setParams(const VectorAD &params);
	virtual void getParams(VectorAD &params);

	/// This function represents the model Xd = B*u
	/// Xd is velocities in body coordinates
	virtual void forward(const VectorAD &u, VectorAD &Xd);
	virtual ADF  loss(const VectorAD &gt_vec, VectorAD &vec);

	virtual void evaluate(const VectorAD &gt_vec, const VectorAD &vec, ADF &ang_err, ADF &lin_err);
	virtual void integrate(const VectorAD &X0, VectorAD &X1);  

private:
	MatrixAD m_params;
	const ADF m_timestep{1e-2};
};
