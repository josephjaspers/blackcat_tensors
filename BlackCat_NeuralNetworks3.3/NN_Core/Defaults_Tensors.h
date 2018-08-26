/*
 * Defaults_Tensors.h
 *
 *  Created on: Jun 28, 2018
 *      Author: joseph
 */

#ifndef DEFAULTS_TENSORS_H_
#define DEFAULTS_TENSORS_H_

namespace BC {
namespace NN {

using scal = BC::Scalar<fp_type, ml>;
using vec = BC::Vector<fp_type, ml>;
using mat = BC::Matrix<fp_type, ml>;
using cube = BC::Cube<fp_type, ml>;

using scal_view = BC::Scalar_View<fp_type, ml>;
using vec_view = BC::Vector_View<fp_type, ml>;
using mat_view = BC::Matrix_View<fp_type, ml>;
using cube_view = BC::Cube_View<fp_type, ml>;

}
}
#endif /* DEFAULTS_TENSORS_H_ */
