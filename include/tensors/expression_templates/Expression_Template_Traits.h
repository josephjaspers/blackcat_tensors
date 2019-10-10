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
namespace tensors {
namespace exprs { 

template<class,class,class> struct Binary_Expression;
template<class,class>		struct Unary_Expression;

namespace detail {

#define BC_QUERY_TAG(tag_name)\
template<class T>\
using query_##tag_name = typename T::tag_name;

BC_QUERY_TAG(system_tag)
BC_QUERY_TAG(value_type)
BC_QUERY_TAG(allocation_type)
BC_QUERY_TAG(copy_assignable)
BC_QUERY_TAG(copy_constructible)
BC_QUERY_TAG(move_assignable)
BC_QUERY_TAG(move_constructible)
BC_QUERY_TAG(requires_greedy_evaluation)
BC_QUERY_TAG(stack_allocated)
BC_QUERY_TAG(optimizer_temporary)
BC_QUERY_TAG(expression_template_expression_type);
BC_QUERY_TAG(expression_template_array_type);

} //end of ns detail

#define BC_TAG_DEFINITION(name, using_name, default_value)\
struct name { using using_name = default_value; };\
namespace detail { 	BC_QUERY_TAG(using_name) }

BC_TAG_DEFINITION(temporary_tag, is_temporary_value, std::true_type);
BC_TAG_DEFINITION(noncontinuous_memory_tag, is_noncontinuous_in_memory, std::true_type);

#undef BC_TAG_DEFINITION
#undef BC_QUERY_TAG


class BC_View {
	using is_view_type = std::true_type;
	using copy_constructible = std::false_type;
	using move_constructible = std::false_type;
    using copy_assignable    = std::true_type;
	using move_assignable    = std::false_type;
};

namespace detail {
template<class T> using query_is_view_type = typename T::is_view_type;
}

template<class T>
struct expression_traits {

	using system_tag = BC::traits::conditional_detected_t<
			detail::query_system_tag, T, host_tag>;

	using allocation_tag = BC::traits::conditional_detected_t<
			detail::query_allocation_type, T, system_tag>;

	using value_type = BC::traits::conditional_detected_t<
			detail::query_value_type, T, void>;

	using is_move_constructible = BC::traits::conditional_detected_t<
			detail::query_move_constructible, T, std::true_type>;

	using is_copy_constructible = BC::traits::conditional_detected_t<
			detail::query_copy_constructible, T, std::true_type>;

	using is_move_assignable = BC::traits::conditional_detected_t<
			detail::query_move_assignable, T, std::true_type>;

	using is_copy_assignable = BC::traits::conditional_detected_t<
			detail::query_copy_assignable, T, std::true_type>;

	using requires_greedy_evaluation = BC::traits::conditional_detected_t<
			detail::query_requires_greedy_evaluation,T, std::false_type>;

	using is_array = BC::traits::conditional_detected_t<
			detail::query_expression_template_array_type, T, std::false_type>;

	using is_expr = BC::traits::conditional_detected_t<
			detail::query_expression_template_expression_type, T, std::false_type>;

	using is_expression_template =
			BC::traits::truth_type<is_array::value || is_expr::value>;

	using is_temporary = BC::traits::conditional_detected_t<
			detail::query_is_temporary_value, T, std::false_type>;

	using is_stack_allocated = BC::traits::conditional_detected_t<
			detail::query_stack_allocated, T, std::false_type>;

	using is_noncontinuous = BC::traits::conditional_detected_t<
			detail::query_is_noncontinuous_in_memory, T, std::false_type>;

	using is_continuous = BC::traits::truth_type<!is_noncontinuous::value>;

};

} //ns BC
} //ns exprs
} //ns tensors

#endif /* BLACKCAT_INTERNAL_FORWARD_DECLS_H_ */
