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
namespace et {

template<class,class,class> class Binary_Expression;
template<class,class>       class Unary_Expression;

namespace bc_type_traits {

template<class T> using allocator_of = std::decay_t<typename T::allocator_t>;
template<class T> using scalar_of    = std::decay_t<typename T::value_type>;

template<class T> constexpr bool is_move_constructible_v= T::move_constructible;
template<class T> constexpr bool is_copy_constructible_v= T::copy_constructible;
template<class T> constexpr bool is_move_assignable_v = T::move_assignable;
template<class T> constexpr bool is_copy_assignable_v = T::copy_assignable;

}
}

//push type_traits into BC_namespace
using namespace et::bc_type_traits;
}


#endif /* BLACKCAT_INTERNAL_FORWARD_DECLS_H_ */
