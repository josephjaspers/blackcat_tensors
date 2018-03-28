/*
 * Defaults.h
 *
 *  Created on: Feb 22, 2018
 *      Author: joseph
 */

#ifndef DEFAULTS_H_
#define DEFAULTS_H_

#include "BlackCat_Tensors.h"
namespace BC {

/*
 * This file defines the default types of tensors and which mathlibrary to use.
 * Currently only CPU lib is supported
 */

using namespace NN_Abreviated_Functions;
struct BASE;
using ml = CPU;
using fp_type = float;
//
//does not work
//using ml = GPU;
//using fp_type = float;
//

using vec = Vector<fp_type, ml>;
using mat = Matrix<fp_type, ml>;
using cube = Cube<fp_type, ml>;
using scal = Scalar<fp_type, ml>;
template<class T, class ML = ml> using mat_expr = Matrix<T, ML>;
template<class T, class ML = ml> using vec_expr = Vector<T, ML>;

}

#endif /* DEFAULTS_H_ */
