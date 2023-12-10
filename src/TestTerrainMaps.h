#pragma once
#include "types/TerrainMap.h"

template<typename Scalar>
class BumpyTerrainMap : public TerrainMap<Scalar>
{
public:
	virtual ~BumpyTerrainMap() = default;
	virtual Scalar getAltitude(const Scalar &x, const Scalar &y) const;
};



template<typename Scalar>
class SlopeTerrainMap : public TerrainMap<Scalar>
{
public:
	virtual ~SlopeTerrainMap() = default;
	virtual Scalar getAltitude(const Scalar &x, const Scalar &y) const;
};



template<typename Scalar>
class FlatTerrainMap : public TerrainMap<Scalar>
{
public:
	virtual ~FlatTerrainMap() = default;
	virtual Scalar getAltitude(const Scalar &x, const Scalar &y) const;
};
