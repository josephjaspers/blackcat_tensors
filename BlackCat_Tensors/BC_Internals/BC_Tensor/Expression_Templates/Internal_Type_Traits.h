/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_INTERNAL_FORWARD_DECLS_H_
#define BLACKCAT_INTERNAL_FORWARD_DECLS_H_

namespace BC {
namespace internal {

template<class,class,class> class Binary_Expression;
template<class,class>		class Unary_Expression;

template<class T> using mathlib_of = std::decay_t<typename T::allocator_t>;
template<class T> using scalar_of  = std::decay_t<typename T::scalar_t>;


template<class T, class enabler=void>
struct BC_array_move_constructible_overrider {
	static constexpr bool boolean = false;
};
template<class T, class enabler=void>
struct BC_array_copy_constructible_overrider {
	static constexpr bool boolean = false;
};
template<class T, class enabler=void>
struct BC_array_move_assignable_overrider {
	static constexpr bool boolean = false;
};
template<class T, class enabler=void>
struct BC_array_copy_assignable_overrider {
	static constexpr bool boolean = false;
};

template<class T, class enabler=void>
struct BC_lvalue_type_overrider {
	static constexpr bool boolean = false;
};


template<class T> constexpr bool BC_array_move_constructible() {
	return BC_array_move_constructible_overrider<T>::boolean;
}
template<class T> constexpr bool BC_array_copy_constructible() {
	return BC_array_copy_constructible_overrider<T>::boolean;
}
template<class T> constexpr bool BC_array_move_assignable() {
	return BC_array_move_assignable_overrider<T>::boolean;
}
template<class T> constexpr bool BC_array_copy_assignable()	{
	return BC_array_copy_assignable_overrider<T>::boolean;
}
template<class T> constexpr bool BC_lvalue()	{
	return BC_lvalue_type_overrider<T>::boolean;
}
template<class T> constexpr bool BC_rvalue()	{
	return !BC_lvalue<T>();
}
}

}
#endif /* BLACKCAT_INTERNAL_FORWARD_DECLS_H_ */
