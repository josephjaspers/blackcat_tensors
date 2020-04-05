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

namespace bc {
namespace tensors {
namespace exprs { 

template<class,class,class> struct Bin_Op;
template<class,class>		struct Un_Op;

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
BC_QUERY_TAG(is_view)
BC_QUERY_TAG(is_const_view)

} //end of ns detail

#define BC_TAG_DEFINITION(name, using_name, default_value)\
struct name { using using_name = default_value; };\
namespace detail { 	BC_QUERY_TAG(using_name) }

BC_TAG_DEFINITION(temporary_tag, is_temporary_value, std::true_type);
BC_TAG_DEFINITION(noncontinuous_memory_tag, is_noncontinuous_in_memory, std::true_type);

#undef BC_TAG_DEFINITION
#undef BC_QUERY_TAG

class BC_View {
	using is_view = std::true_type;
	using copy_constructible = std::false_type;
	using move_constructible = std::false_type;
    using copy_assignable    = std::true_type;
	using move_assignable    = std::false_type;
};

class BC_Const_View {
	using is_view = std::true_type;
	using is_const_view = std::true_type;
	using copy_constructible = std::false_type;
	using move_constructible = std::false_type;
	using copy_assignable    = std::false_type;
	using move_assignable    = std::false_type;
};

namespace detail {
template<class T> using query_is_view_type = typename T::is_view_type;
}

template<class T>
struct expression_traits {

	using system_tag = bc::traits::conditional_detected_t<
			detail::query_system_tag, T, host_tag>;

	using allocation_tag = bc::traits::conditional_detected_t<
			detail::query_allocation_type, T, system_tag>;

	using value_type = bc::traits::conditional_detected_t<
			detail::query_value_type, T, void>;

	using is_const_view = bc::traits::conditional_detected_t<
			detail::query_is_const_view, T, std::false_type>;

	using is_view = bc::traits::conditional_detected_t<
			detail::query_is_view, T, is_const_view>;

	using is_move_constructible = bc::traits::conditional_detected_t<
			detail::query_move_constructible, T, std::true_type>;

	using is_copy_constructible = bc::traits::conditional_detected_t<
			detail::query_copy_constructible, T, std::true_type>;

	using is_move_assignable = bc::traits::conditional_detected_t<
			detail::query_move_assignable, T, std::true_type>;

	using is_copy_assignable = bc::traits::conditional_detected_t<
			detail::query_copy_assignable, T, std::true_type>;

	using is_blas_expression = std::is_base_of<bc::oper::BLAS_Function, T>;

	using requires_greedy_evaluation = bc::traits::conditional_detected_t<
			detail::query_requires_greedy_evaluation,T, std::false_type>;

	using is_array = bc::traits::conditional_detected_t<
			detail::query_expression_template_array_type, T, std::false_type>;

	using is_expr = bc::traits::conditional_detected_t<
			detail::query_expression_template_expression_type, T, std::false_type>;

	using is_expression_template =
			bc::traits::truth_type<is_array::value || is_expr::value>;

	using is_temporary = bc::traits::conditional_detected_t<
			detail::query_is_temporary_value, T, std::false_type>;

	using is_stack_allocated = bc::traits::conditional_detected_t<
			detail::query_stack_allocated, T, std::false_type>;

	using is_noncontinuous = bc::traits::conditional_detected_t<
			detail::query_is_noncontinuous_in_memory, T, std::false_type>;

	using is_continuous = bc::traits::truth_type<!is_noncontinuous::value>;

};

} //ns BC
} //ns exprs
} //ns tensors

#endif /* BLACKCAT_INTERNAL_FORWARD_DECLS_H_ */
