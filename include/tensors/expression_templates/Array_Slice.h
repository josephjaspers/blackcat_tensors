/*
 * Array_Slice.h
 *
 *  Created on: Dec 24, 2018
 *      Author: joseph
 */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_SLICE_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_SLICE_H_

#include "Expression_Template_Base.h"
#include "Array_Kernel_Array.h"
#include "Array.h"

namespace BC {
namespace tensors {
namespace exprs {

template<class Shape, class ValueType, class Allocator, class... Tags>
struct Array_Slice:
		Kernel_Array<
				Shape,
				ValueType,
				typename BC::allocator_traits<Allocator>::system_tag,
				Tags...> {

	using value_type = ValueType;
	using allocator_type = Allocator;
	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;
	using stream_type = BC::Stream<system_tag>;
	using parent = Kernel_Array<Shape, ValueType, system_tag, Tags...>;
	using move_assignable = std::false_type;

	stream_type m_stream;
	allocator_type m_allocator;

	template<class... Args> BCHOT
	Array_Slice(stream_type stream, allocator_type allocator, Args... args):
			parent(args...),
			m_stream(stream),
			m_allocator(allocator) {}

	const allocator_type& get_allocator() const { return m_allocator; }
	      allocator_type& get_allocator()       { return m_allocator; }

	const stream_type& get_stream() const { return m_stream; }
	      stream_type& get_stream()       { return m_stream; }
};

namespace {

template<int Dimension, class Parent, class... Tags>
using slice_type_from_parent = Array_Slice<
		BC::Shape<BC::traits::max(Dimension,0)>,
		typename Parent::value_type,
		typename Parent::allocator_type,
		BC_View, Tags...>;

template<int Dimension, class Parent, class... Tags>
using strided_slice_type_from_parent = Array_Slice<
		std::enable_if_t<Dimension==1, Strided_Vector_Shape>,
		typename Parent::value_type,
		typename Parent::allocator_type,
		BC_View, Tags...>;
}

template<class Parent>
auto make_row(Parent& parent, BC::size_t index) {
	using slice_type = strided_slice_type_from_parent<1, Parent, noncontinuous_memory_tag>;
	return slice_type(
			parent.get_stream(),
			parent.get_allocator(),
			Strided_Vector_Shape(parent.cols(), parent.leading_dimension(1)),
			parent.data() + index);
}

template<class Parent>
auto make_diagnol(Parent& parent, BC::size_t diagnol_index) {

	BC::print(__func__);
    BC::size_t stride = parent.leading_dimension(1) + 1;
    BC::size_t length = BC::traits::min(parent.rows(), parent.cols() - diagnol_index);
    BC::size_t ptr_index = diagnol_index > 0 ? parent.leading_dimension(1) * diagnol_index : std::abs(diagnol_index);

    BC::print(parent.rows(), parent.cols());
    BC::print(stride, length);

    using slice_type = strided_slice_type_from_parent<1, Parent, noncontinuous_memory_tag>;
	return slice_type(parent.get_stream(),
						parent.get_allocator(),
						Strided_Vector_Shape(length, stride),
						parent.data() + ptr_index);
}

template<class Parent, class=std::enable_if_t<!expression_traits<Parent>::is_continuous::value>>
static auto make_slice(Parent& parent, BC::size_t index) {
	using slice_type = slice_type_from_parent<BC::traits::max(0,Parent::tensor_dimension-1), Parent, noncontinuous_memory_tag>;
	using scalar_type = slice_type_from_parent<0, Parent>;
	return std::conditional_t<Parent::tensor_dimension == 1, scalar_type, slice_type>(
			parent.get_stream(),
			parent.get_allocator(),
			parent.get_shape(),
			parent.data() + parent.slice_ptr_index(index));
}

template<class Parent, class=std::enable_if_t<expression_traits<Parent>::is_continuous::value>, int differentiator=0>
static auto make_slice(Parent& parent, BC::size_t index) {

	using slice_type = slice_type_from_parent<BC::traits::max(0,Parent::tensor_dimension-1), Parent>;
	using scalar_type = slice_type_from_parent<0, Parent>;
	return std::conditional_t<Parent::tensor_dimension == 1, scalar_type, slice_type>(
			parent.get_stream(),
			parent.get_allocator(),
			parent.get_shape(),
			parent.data() + parent.slice_ptr_index(index));
}
template<class Parent>
static auto make_ranged_slice(Parent& parent, BC::size_t from, BC::size_t to) {
	BC::size_t range = to - from;
	BC::size_t index = parent.slice_ptr_index(from);

	BC::Dim<Parent::tensor_dimension> inner_shape = parent.inner_shape();
	inner_shape[Parent::tensor_dimension-1] = range;

	using slice_type = slice_type_from_parent<Parent::tensor_dimension, Parent>;
	return slice_type(parent.get_stream(),
						parent.get_allocator(),
						BC::Shape<Parent::tensor_dimension>(inner_shape),
						parent.data() + index);
}

template<class Parent, class ShapeLike>
static auto make_view(Parent& parent, ShapeLike shape) {
	using slice_type = slice_type_from_parent<ShapeLike::tensor_dimension, Parent>;
	return slice_type(parent.get_stream(),
						parent.get_allocator(),
						BC::Shape<ShapeLike::tensor_dimension>(shape),
						parent.data());
}

template<class Parent>
static auto make_scalar(Parent& parent, BC::size_t index) {
	using slice_type = slice_type_from_parent<0, Parent>;
	return slice_type(parent.get_stream(),
						parent.get_allocator(),
						BC::Shape<0>(),
						parent.data() + index);
}

template<class Parent, class ShapeLike>
auto make_chunk(Parent& parent,
			BC::Dim<Parent::tensor_dimension> index_points,
			ShapeLike shape) {
	static_assert(ShapeLike::tensor_dimension > 1,
			"TENSOR CHUNKS MUST HAVE DIMENSIONS GREATER THAN 1, USE SCALAR OR RANGED_SLICE OTHERWISE");

	using slice_type = slice_type_from_parent<ShapeLike::tensor_dimension, Parent, noncontinuous_memory_tag>;
	return slice_type(parent.get_stream(),
						parent.get_allocator(),
						Shape<ShapeLike::tensor_dimension>(shape, parent.get_shape()),
						parent.data() + parent.dims_to_index(index_points));
}


} //ns BC
} //ns exprs
} //ns tensors



#endif /* Array_Slice_H_ */
