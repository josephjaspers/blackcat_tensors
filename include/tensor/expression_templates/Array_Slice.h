/*
 * Array_Slice.h
 *
 *  Created on: Dec 24, 2018
 *      Author: joseph
 */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_SLICE_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_SLICE_H_

#include "Common.h"
#include "Array_Base.h"
#include "Shape.h"


namespace BC {
namespace exprs {

template<class Parent, class ViewType>
struct Array_Slice : ViewType {

	using allocator_t 	 = typename Parent::allocator_t;
	using context_t 	 = typename Parent::context_t;

private:
	template<class T>
	using const_if_parent_is_const = std::conditional_t<std::is_const<Parent>::value, const T, T>;


	using m_allocator_t  = const_if_parent_is_const<allocator_t>;
	using m_context_t 	 = const_if_parent_is_const<context_t>;

public:

	m_context_t m_context;
	m_allocator_t& m_allocator;

	template<class... Args>
	BCHOT Array_Slice(Parent& parent_, Args... args_)
	: ViewType(args_...),
	  m_context(parent_.get_context()),
	  m_allocator(parent_.get_allocator()) {}

	BCHOT Array_Slice(const Array_Slice&) = default;
	BCHOT Array_Slice(Array_Slice&&) = default;

	const m_allocator_t& get_allocator() const { return m_allocator; }
		  m_allocator_t& get_allocator() 	   { return m_allocator; }

	const auto& internal_base() const { return *this; }
		  auto& internal_base() 	  { return *this; }

	const m_context_t& get_context() const  { return m_context; }
		  m_context_t& get_context()  		{ return m_context; }
};


template<class Parent>
auto make_row(Parent parent, BC::size_t index) {
	using value_type = BC::meta::propagate_const_t<Parent, typename Parent::value_type>;
	using expression_template = ArrayExpression<1, value_type, typename Parent::system_tag, BC_View, BC_Noncontinuous>;
	return Array_Slice<Parent, expression_template>(parent,
			Shape<1>(parent.cols(), parent.leading_dimension(0) + 1), parent.memptr() + index);
}

template<class Parent>
auto make_diagnol(Parent parent, BC::size_t diagnol_index) {
    BC::size_t stride = parent.leading_dimension(0) + 1;
    BC::size_t length = BC::meta::min(parent.rows(), parent.cols() - diagnol_index);
    BC::size_t ptr_index = diagnol_index > 0 ? parent.leading_dimension(0) * diagnol_index : std::abs(diagnol_index);
	using expression_template = ArrayExpression<1,
			typename Parent::value_type,
			typename Parent::system_tag,
			BC_View, BC_Noncontinuous>;

	return Array_Slice<Parent, expression_template>(parent,
			Shape<1>(length, stride), parent.memptr() + ptr_index);
}


template<class Parent, class=std::enable_if_t<!expression_traits<Parent>::is_continuous>>
static auto make_slice(Parent& parent, BC::size_t index) {
	static constexpr int Dimension = Parent::DIMS-1;
	using value_type = BC::meta::propagate_const_t<Parent, typename Parent::value_type>;
	using expression_template = ArrayExpression<Dimension, value_type, typename Parent::system_tag, BC_View, BC_Noncontinuous>;
	return Array_Slice<Parent, expression_template>(parent,
			parent.get_shape(), parent.memptr() + parent.slice_ptr_index(index));
}
template<class Parent, class=std::enable_if_t<expression_traits<Parent>::is_continuous>, int differentiator=0>
static auto make_slice(Parent& parent, BC::size_t index) {
	static constexpr int Dimension = Parent::DIMS-1;
	using value_type = BC::meta::propagate_const_t<Parent, typename Parent::value_type>;
	using expression_template = ArrayExpression<Dimension, value_type, typename Parent::system_tag, BC_View>;
	return Array_Slice<Parent, expression_template>(parent,
			parent.get_shape(), parent.memptr() + parent.slice_ptr_index(index));
}
template<class Parent>
static auto make_ranged_slice(Parent& parent, BC::size_t from, BC::size_t to) {
	constexpr BC::size_t dim_id = Parent::DIMS;
	BC::size_t range = to - from;
	BC::size_t index = parent.slice_ptr_index(from);

	BC::array<dim_id, BC::size_t> inner_shape = parent.inner_shape();

	inner_shape[dim_id-1] = range;
	BC::exprs::Shape<dim_id> new_shape(inner_shape);

	static constexpr int Dimension = Parent::DIMS;
	using value_type = BC::meta::propagate_const_t<Parent, typename Parent::value_type>;
	using expression_template = ArrayExpression<Dimension, value_type, typename Parent::system_tag>;

	return Array_Slice<Parent, expression_template>(parent,
			new_shape, parent.memptr() + index);
}

template<class Parent, int ndims>
static auto make_view(Parent& parent, BC::array<ndims, BC::size_t> shape) {
	using value_type = BC::meta::propagate_const_t<Parent, typename Parent::value_type>;
	using expression_template = ArrayExpression<ndims, value_type, typename Parent::system_tag>;
	return Array_Slice<Parent, expression_template>(parent,
			BC::Shape<ndims>(shape), parent.memptr());
}

template<class Parent>
	static auto make_scalar(Parent& parent, BC::size_t index) {
		using value_type = BC::meta::propagate_const_t<Parent, typename Parent::value_type>;
		using system_tag = typename Parent::system_tag;

		return Array_Slice<Parent, ArrayExpression<0, value_type, system_tag>>(
				parent, BC::Shape<0>(), parent.memptr() + index);
	}

template<class Parent, int ndims>
auto make_chunk(Parent& parent, BC::array<Parent::DIMS, int> index_points, BC::array<ndims, int> shape) {
	static_assert(ndims > 1, "TENSOR CHUNKS MUST HAVE DIMENSIONS GREATER THAN 1, USE SCALAR OR RANGED_SLICE OTHERWISE");
	BC::size_t index = parent.dims_to_index(index_points);
	SubShape<ndims> chunk_shape = SubShape<ndims>(shape, parent.get_shape());
	using value_type = BC::meta::propagate_const_t<Parent, typename Parent::value_type>;
	using expression_template = ArrayExpression<ndims, value_type, typename Parent::system_tag, BC_Noncontinuous, BC_View>;
	return Array_Slice<Parent, expression_template>(parent,
			chunk_shape, parent.memptr() + index);
}


}
}


#endif /* Array_Slice_H_ */
