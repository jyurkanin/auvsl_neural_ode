#pragma once

//#include <cppad/cg.hpp>
#include <cppad/cppad.hpp>

//auto-derivative types. Pure psychosis.

// typedef CppAD::cg::CG<double> CGF;
// typedef CppAD::AD<CGF> ADCF;
// typedef CppAD::AD<ADCF> ADAD;
typedef CppAD::AD<double> ADF;
