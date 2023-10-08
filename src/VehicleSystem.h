#pragma once

#include "types/System.h"
#include "types/SystemFactory.h"
#include "types/TerrainMap.h"
#include "HybridDynamics.h"

// I believe this is ready.
// This represents an ODE and loss function

template<typename Scalar>
class VehicleSystem : public System<Scalar>
{
public:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorS;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixS;
  
	VehicleSystem(const std::shared_ptr<const TerrainMap<Scalar>> &map);
	~VehicleSystem();
	
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
	MatrixS m_params;
	HybridDynamics m_hybrid_dynamics;
};


template<typename Scalar>
class VehicleSystemFactory : public SystemFactory<Scalar>
{
public:
	VehicleSystemFactory(const std::shared_ptr<const TerrainMap<Scalar>>& map) : SystemFactory<Scalar>(map)
	{
		
	}
	
	virtual std::shared_ptr<System<Scalar>> makeSystem()
	{
		return std::make_shared<VehicleSystem<Scalar>>(this->m_terrain_map);
	}
};
