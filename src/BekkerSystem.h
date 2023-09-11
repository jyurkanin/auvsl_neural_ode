#pragma once

#include "types/System.h"
#include "types/SystemFactory.h"
#include "BekkerDynamics.h"

template<typename Scalar>
class BekkerSystem : public System<Scalar>
{
public:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;

	BekkerSystem();
	~BekkerSystem();

	virtual void   getDefaultParams(VectorS &params);
	virtual void   getDefaultInitialState(VectorS &state);
	virtual void   setParams(const VectorS &params);
	virtual void   getParams(VectorS &params);
	virtual void   forward(const VectorS &X, VectorS &Xd);
	virtual Scalar loss(const VectorS &gt_vec, VectorS &vec);

	virtual void evaluate(const VectorS &gt_vec, const VectorS &vec, Scalar &ang_err, Scalar &lin_err);
	virtual void integrate(const VectorS &X0, VectorS &X1);
	virtual VectorS initializeState(const GroundTruthDataRow &gt_state);
	
private:
	BekkerDynamics m_bekker_dynamics;
};

template<typename Scalar>
class BekkerSystemFactory : public SystemFactory<Scalar>
{
public:
	virtual std::shared_ptr<System<Scalar>> makeSystem()
	{
		return std::make_shared<BekkerSystem<Scalar>>();
	}
}
