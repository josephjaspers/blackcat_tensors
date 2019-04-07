/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_INTERNAL_FORWARD_DECLS_H_
#define BLACKCAT_INTERNAL_FORWARD_DECLS_H_

#include <type_traits>


namespace BC {
namespace exprs {

class BC_Type {}; //a type inherited by expressions and tensor_cores, it is used a flag and lacks a "genuine" implementation
class BC_Array {};
class BC_Expr  {};
class BC_Temporary {};
class BC_Scalar_Constant {};
class BC_Constant {};

template<class,class,class> class Binary_Expression;
template<class,class>       class Unary_Expression;

template<class T> using allocator_of  = typename T::allocator_t;
template<class T> using scalar_of     = typename T::value_type;


template<class T>
struct expression_traits {

	static constexpr bool is_move_constructible = T::move_constructible;
	static constexpr bool is_copy_constructible = T::copy_constructible;
	static constexpr bool is_move_assignable 	= T::move_assignable;
	static constexpr bool is_copy_assignable 	= T::copy_assignable;

	 static constexpr bool is_bc_type  	= std::is_base_of<BC_Type, T>::value;
	 static constexpr bool is_array  	= std::is_base_of<BC_Array, T>::value;
	 static constexpr bool is_expr  	= std::is_base_of<BC_Expr, T>::value;
	 static constexpr bool is_temporary = std::is_base_of<BC_Temporary, T>::value;
	 static constexpr bool is_constant  = std::is_base_of<BC_Constant, T>::value;
};


}

}


#endif /* BLACKCAT_INTERNAL_FORWARD_DECLS_H_ */
