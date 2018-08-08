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

using Scalar = BC::Scalar<fp_type, ml>;
using Vector = BC::Vector<fp_type, ml>;
using Matrix = BC::Matrix<fp_type, ml>;

using Cube = BC::Cube<fp_type, ml>;
//using Scalar = BC::Scalar<fp_type, ml>;
//using Scalar = BC::Scalar<fp_type, ml>;
//
using scal 		= Scalar;
using vec 		= Vector;
using mat 		= Matrix;
using cube 		= Cube;
//using tensor4 	= Tensor4<fp_type, ml>;
//using tensor5 	= Tensor5<fp_type, ml>;

//template<class T> using f_scal 		= Scalar<T, ml>;
//template<class T> using f_vec 		= Vector<T, ml>;
//template<class T> using f_mat 		= Matrix<T, ml>;
//template<class T> using f_cube 		= Cube<T, ml>;
//template<class T> using f_tensor4 	= Tensor4<T, ml>;
//template<class T> using f_tensor5 	= Tensor5<T, ml>;

//template<int> struct dim_of_tensor;
//template<> struct dim_of_tensor<0> { using type = scal; 	};
//template<> struct dim_of_tensor<1> { using type = vec; 		};
//template<> struct dim_of_tensor<2> { using type = mat; 		};
//template<> struct dim_of_tensor<3> { using type = cube; 	};
//template<> struct dim_of_tensor<4> { using type = tensor4; 	};
//template<> struct dim_of_tensor<5> { using type = tensor5; 	};
}
}
#endif /* DEFAULTS_TENSORS_H_ */
