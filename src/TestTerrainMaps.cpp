#include "TestTerrainMaps.h"
#include "types/Scalars.h"


template<typename Scalar>
Scalar FlatTerrainMap<Scalar>::getAltitude(const Scalar &x, const Scalar &y) const
{
	return Scalar{0.0};
}

template<typename Scalar>
Scalar SlopeTerrainMap<Scalar>::getAltitude(const Scalar &x, const Scalar &y) const
{
	return 0.1*x;
}

template<typename Scalar>
Scalar BumpyTerrainMap<Scalar>::getAltitude(const Scalar &x, const Scalar &y) const
{
	Scalar zero{0.0};
	Scalar temp = 0.1*CppAD::sin(x);
	return CppAD::CondExpGt(temp, zero, temp, zero);
}





template class FlatTerrainMap<ADF>;
template class SlopeTerrainMap<ADF>;
template class BumpyTerrainMap<ADF>;

