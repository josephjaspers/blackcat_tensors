/*
 * Tags.h
 *
 *  Created on: Feb 11, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_TAGS_H_
#define BLACKCAT_TENSORS_TAGS_H_

/*
 * This file should be included first,
 * All tags in the BC namespace should be defined here.
 */

namespace BC {

class host_tag;
class device_tag;

class BC_Type {}; //a type inherited by expressions and tensor_cores, it is used a flag and lacks a "genuine" implementation
class BC_Array {};
class BC_Expr  {};
class BC_Temporary {};
class BLAS_Function {};
class BC_Scalar_Constant {};
class BC_Constant {};

template<class T> static constexpr bool is_bc_type()   { return std::is_base_of<BC_Type, T>::value; }
template<class T> static constexpr bool is_array()     { return std::is_base_of<BC_Array, T>::value; }
template<class T> static constexpr bool is_expr()      { return std::is_base_of<BC_Expr, T>::value; }
template<class T> static constexpr bool is_temporary() { return std::is_base_of<BC_Temporary, T>::value; }
template<class T> static constexpr bool is_blas_func() { return std::is_base_of<BLAS_Function, T>::value; }
template<class T> static constexpr bool is_constant()  { return std::is_base_of<BC_Constant, T>::value; }

}

#endif /* TAGS_H_ */
