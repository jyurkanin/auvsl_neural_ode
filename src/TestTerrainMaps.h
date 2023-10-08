#pragma once
#include "types/TerrainMap.h"

template<typename Scalar>
class BumpyTerrainMap : public TerrainMap<Scalar>
{
public:
	virtual Scalar getAltitude(const Scalar &x, const Scalar &y) const;
};



template<typename Scalar>
class SlopeTerrainMap : public TerrainMap<Scalar>
{
public:
	virtual Scalar getAltitude(const Scalar &x, const Scalar &y) const;
};



template<typename Scalar>
class FlatTerrainMap : public TerrainMap<Scalar>
{
public:
	virtual Scalar getAltitude(const Scalar &x, const Scalar &y) const;
};
