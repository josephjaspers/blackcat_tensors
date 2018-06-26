/*
 * Defaults.h
 *
 *  Created on: Feb 22, 2018
 *      Author: joseph
 */

#ifndef GPU_DEFAULTS_H_
#define GPU_DEFAULTS_H_

#include "BlackCat_Tensors.h"
namespace BC {
namespace NN {

/*
 * This file defines the default types of tensors and which mathlibrary to use.
 * Currently only CPU lib is supported
 */
struct BASE;

using ml = BC::GPU;
using fp_type = float;

using scal = Scalar<fp_type, ml>;
using vec = Vector<fp_type, ml>;
using mat = Matrix<fp_type, ml>;
using cube = Cube<fp_type, ml>;
using tensor4 = Tensor4<fp_type, ml>;
using tensor5 = Tensor5<fp_type, ml>;

template<class T> using f_scal = Scalar<T, ml>;
template<class T> using f_vec = Vector<T, ml>;
template<class T> using f_mat = Matrix<T, ml>;
template<class T> using f_cube = Cube<T, ml>;
template<class T> using f_tensor4 = Tensor4<T, ml>;
template<class T> using f_tensor5 = Tensor5<T, ml>;

}
}

#endif /* DEFAULTS_H_ */
