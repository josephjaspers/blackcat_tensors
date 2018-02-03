/*
 * Defaults.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef DEFAULTS_CU_
#define DEFAULTS_CU_

#include "BlackCat_Tensors.h"

namespace BC {


enum LAYER_TYPE  {
	FeedForward_ = 0,
};
//#define DEFAULT_DEVICE_GPU
#ifdef DEFAULT_DEVICE_GPU
#define BLACKCAT_DEFAULT_MATHLIB_GPU //this changes the default math lib of the BlackCatTensors
using fp_type =  float;
using ml = GPU;
#else
using fp_type = double;
using ml = CPU;
#endif

//default classes
using mat = Matrix<fp_type, ml>;
using vec = Vector<fp_type, ml>;

//These are the expression classes
template<class T, class ML> using mat_expr = Matrix<T, ML>;
template<class T, class ML> using vec_expr = Vector<T, ML>;


}
#endif /* DEFAULTS_CU_ */
