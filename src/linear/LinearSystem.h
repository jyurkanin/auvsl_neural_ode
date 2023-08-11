#pragma once

#include <cpp_bptt.h>

// I believe this is ready.
// This represents an ODE and loss function

template<typename Scalar>
class LinearSystem : public cpp_bptt::System<Scalar>
{
public:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;
	
	LinearSystem();
	~LinearSystem();

	virtual void   getDefaultParams(VectorS &params);
	virtual void   getDefaultInitialState(VectorS &state);
	virtual void   setParams(const VectorS &params);
	virtual void   getParams(VectorS &params);

	/// This function represents the model Xd = B*u
	/// Xd is velocities in body coordinates
	virtual void   forward(const VectorS &u, VectorS &Xd);
	virtual Scalar loss(const VectorS &gt_vec, VectorS &vec);

	void evaluate(const VectorS &gt_vec, const VectorS &vec, Scalar &ang_err, Scalar &lin_err);
	void integrate(const VectorS &X0, VectorS &X1);  
	
	MatrixS m_params;
};
