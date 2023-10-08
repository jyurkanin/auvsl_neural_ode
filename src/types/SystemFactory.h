#pragma once

#include "types/System.h"
#include "types/TerrainMap.h"

#include <memory>

template<typename Scalar>
class SystemFactory
{
public:
	SystemFactory(const std::shared_ptr<const TerrainMap<Scalar>> &map) : m_terrain_map{map} {}
	virtual std::shared_ptr<System<Scalar>> makeSystem() = 0;
	
protected:
	const std::shared_ptr<const TerrainMap<Scalar>> m_terrain_map;
};
