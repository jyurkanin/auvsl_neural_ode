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
	Scalar max{1.6};
	Scalar arg = x - 2.0;
	arg = CppAD::CondExpLt(arg, max, arg, zero);
	arg = CppAD::CondExpGt(arg, zero, arg, zero);
	
	Scalar temp = .2*CppAD::sin(arg);
	return CppAD::CondExpGt(temp, zero, temp, zero);
}





template class FlatTerrainMap<ADF>;
template class SlopeTerrainMap<ADF>;
template class BumpyTerrainMap<ADF>;

template class FlatTerrainMap<double>;
template class SlopeTerrainMap<double>;
template class BumpyTerrainMap<double>;

