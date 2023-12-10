#pragma once

#include "memory"

template<typename Scalar>
class TerrainMap
{
public:
	virtual ~TerrainMap() = default;
	virtual Scalar getAltitude(const Scalar &x, const Scalar &y) const = 0;
};
