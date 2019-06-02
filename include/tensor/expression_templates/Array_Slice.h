/*
 * Array_Slice.h
 *
 *  Created on: Dec 24, 2018
 *      Author: joseph
 */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_SLICE_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_SLICE_H_

#include "Expression_Template_Base.h"
#include "Shape.h"

namespace BC {
namespace exprs {

template<int Dimensions, class ValueType, class Allocator, class... Tags>
struct Array_Slice : ArrayExpression<Dimensions, ValueType, typename BC::allocator_traits<Allocator>::system_tag, Tags...> {

	using allocator_t 	 = Allocator;
	using stream_t 	 = BC::Stream<typename BC::allocator_traits<Allocator>::system_tag>;
	using parent = ArrayExpression<Dimensions, ValueType, typename BC::allocator_traits<Allocator>::system_tag, Tags...>;

public:

	stream_t m_stream;
	allocator_t m_allocator;

	template<class... Args>
	BCHOT Array_Slice(stream_t stream_, allocator_t allocator_, Args... args_)
	: parent(args_...),
	  m_stream(stream_),
	  m_allocator(allocator_) {}

	BCHOT Array_Slice(const Array_Slice&) = default;
	BCHOT Array_Slice(Array_Slice&&) = default;

	const allocator_t& get_allocator() const { return m_allocator; }
		  allocator_t& get_allocator() 	   { return m_allocator; }

	const stream_t& get_stream() const  { return m_stream; }
		  stream_t& get_stream()  		{ return m_stream; }
};


template<class Parent>
auto make_row(Parent& parent, BC::size_t index) {
	using slice_type = Array_Slice<1,
			typename Parent::value_type,
			typename Parent::allocator_t,
			BC_View,
			BC_Noncontinuous>;

	return slice_type(parent.get_stream(), parent.get_allocator(),
						Shape<1>(parent.cols(), parent.leading_dimension(0)),
						parent.memptr() + index);
}

template<class Parent>
auto make_diagnol(Parent& parent, BC::size_t diagnol_index) {
    BC::size_t stride = parent.leading_dimension(0) + 1;
    BC::size_t length = BC::meta::min(parent.rows(), parent.cols() - diagnol_index);
    BC::size_t ptr_index = diagnol_index > 0 ? parent.leading_dimension(0) * diagnol_index : std::abs(diagnol_index);

    using slice_type = Array_Slice<1,
    					typename Parent::value_type,
    					typename Parent::allocator_t,
    					BC_View, BC_Noncontinuous>;

	return slice_type(parent.get_stream(),
						parent.get_allocator(),
						Shape<1>(length, stride),
						parent.memptr() + ptr_index);
}


template<class Parent, class=std::enable_if_t<!expression_traits<Parent>::is_continuous>>
static auto make_slice(Parent& parent, BC::size_t index) {
	using slice_type = Array_Slice<Parent::tensor_dimension-1,
						typename Parent::value_type,
						typename Parent::allocator_t,
						BC_View,
						BC_Noncontinuous>;

	return slice_type(parent.get_stream(),
						parent.get_allocator(),
						parent.get_shape(),
						parent.memptr() + parent.slice_ptr_index(index));
}
template<class Parent, class=std::enable_if_t<expression_traits<Parent>::is_continuous>, int differentiator=0>
static auto make_slice(Parent& parent, BC::size_t index) {

	using slice_type = Array_Slice<Parent::tensor_dimension-1,
						typename Parent::value_type,
						typename Parent::allocator_t,
						BC_View>;

	return slice_type(parent.get_stream(),
						parent.get_allocator(),
						parent.get_shape(),
						parent.memptr() + parent.slice_ptr_index(index));
}
template<class Parent>
static auto make_ranged_slice(Parent& parent, BC::size_t from, BC::size_t to) {
	BC::size_t range = to - from;
	BC::size_t index = parent.slice_ptr_index(from);

	BC::array<Parent::tensor_dimension, BC::size_t> inner_shape = parent.inner_shape();
	inner_shape[Parent::tensor_dimension-1] = range;

	using slice_type = Array_Slice<Parent::tensor_dimension,
						typename Parent::value_type,
						typename Parent::allocator_t>;

	return slice_type(parent.get_stream(),
						parent.get_allocator(),
						BC::Shape<Parent::tensor_dimension>(inner_shape),
						parent.memptr() + index);
}

template<class Parent, int Dimension>
static auto make_view(Parent& parent, BC::array<Dimension, BC::size_t> shape) {
	using slice_type = Array_Slice<Dimension,
						typename Parent::value_type,
						typename Parent::allocator_t>;

	return slice_type(parent.get_stream(),
						parent.get_allocator(),
						BC::Shape<Dimension>(shape),
						parent.memptr());
}

template<class Parent>
static auto make_scalar(Parent& parent, BC::size_t index) {
	using slice_type = Array_Slice<0,
						typename Parent::value_type,
						typename Parent::allocator_t>;

	return slice_type(parent.get_stream(),
						parent.get_allocator(),
						BC::Shape<0>(),
						parent.memptr() + index);
}

template<class Parent, int Dimension>
auto make_chunk(Parent& parent, BC::array<Parent::tensor_dimension, BC::size_t> index_points, BC::array<Dimension, int> shape) {
	static_assert(Dimension > 1, "TENSOR CHUNKS MUST HAVE DIMENSIONS GREATER THAN 1, USE SCALAR OR RANGED_SLICE OTHERWISE");

	using slice_type = Array_Slice<
			Dimension,
			typename Parent::value_type,
			typename Parent::allocator_t,
			BC_Noncontinuous,
			BC_View>;

	return slice_type(parent.get_stream(),
						parent.get_allocator(),
						SubShape<Dimension>(shape, parent.get_shape()),
						parent.memptr() + parent.dims_to_index(index_points));
}


}
}


#endif /* Array_Slice_H_ */
