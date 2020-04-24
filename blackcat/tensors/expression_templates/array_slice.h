/*
 * Array_Slice.h
 *
 *  Created on: Dec 24, 2018
 *	  Author: joseph
 */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_SLICE_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_SLICE_H_

#include "expression_template_base.h"
#include "array_kernel_array.h"
#include "array.h"

namespace bc {
namespace tensors {
namespace exprs {

template<class Shape, class ValueType, class Allocator, class... Tags>
struct Array_Slice:
		Kernel_Array<
			Shape,
			ValueType,
			typename bc::allocator_traits<Allocator>::system_tag,
			Tags...>
{
	using system_tag = typename bc::allocator_traits<Allocator>::system_tag;
	using value_type = ValueType;
	using allocator_type = Allocator;
	using stream_type = bc::Stream<system_tag>;
	using shape_type = Shape;

private:

	using parent = Kernel_Array<Shape, ValueType, system_tag, Tags...>;

	stream_type m_stream;
	allocator_type m_allocator;

public:

	using move_assignable = std::false_type;
	using copy_assignable = bc::traits::not_type<bc::traits::sequence_contains_v<BC_Const_View, Tags...>>;

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

template<class Shape, class Parent, class... Tags>
using slice_type_factory = Array_Slice<
		Shape,
		typename Parent::value_type,
		typename Parent::allocator_type,
		std::conditional_t<
			std::is_const<Parent>::value ||
			expression_traits<Parent>::is_const_view::value,
		BC_Const_View,
		BC_View>,
		Tags...>;

template<int Dimension, class Parent, class... Tags>
using slice_type_from_parent = slice_type_factory<
		bc::Shape<bc::traits::max(Dimension,0)>,
		Parent,
		Tags...>;

template<int Dimension, class Parent, class... Tags>
using strided_slice_type_from_parent = slice_type_factory<
		std::enable_if_t<Dimension==1, Strided_Vector_Shape>,
		Parent,
		Tags...>;
}

template<class Parent>
auto make_row(Parent& parent, bc::size_t index) {
	using slice_type = strided_slice_type_from_parent<
		1, Parent, noncontinuous_memory_tag>;

	return slice_type(
		parent.get_stream(),
		parent.get_allocator(),
		Strided_Vector_Shape(parent.cols(), parent.leading_dim(1)),
		parent.data() + index);
}

template<class Parent>
auto make_diagnol(Parent& parent, bc::size_t diagnol_index) {
	bc::size_t stride = parent.leading_dim(1) + 1;
	bc::size_t length = bc::traits::min(
		parent.rows(), parent.cols() - diagnol_index);

	bc::size_t ptr_index = diagnol_index > 0
		? parent.leading_dim(1) * diagnol_index
		: std::abs(diagnol_index);

	using slice_type = strided_slice_type_from_parent<
				1, Parent, noncontinuous_memory_tag>;

	return slice_type(
		parent.get_stream(),
		parent.get_allocator(),
		Strided_Vector_Shape(length, stride),
		parent.data() + ptr_index);
}

template<
	class Parent,
	class=std::enable_if_t<
		!expression_traits<Parent>::is_continuous::value>>
static auto make_slice(Parent& parent, bc::size_t index)
{
	using scalar_type = slice_type_from_parent<0, Parent>;
	using slice_type = slice_type_from_parent<
		bc::traits::max(0,Parent::tensor_dim-1),
		Parent,
		noncontinuous_memory_tag>;

	using slice_t = std::conditional_t<
		Parent::tensor_dim == 1, scalar_type, slice_type>;

	return slice_t(
		parent.get_stream(),
		parent.get_allocator(),
		parent.get_shape(),
		parent.data() + parent.leading_dim() * index);
}

template<
	class Parent,
	class=std::enable_if_t<
			expression_traits<Parent>::is_continuous::value>,
	int differentiator=0>
static auto make_slice(Parent& parent, bc::size_t index)
{
	using scalar_type = slice_type_from_parent<0, Parent>;
	using slice_type = slice_type_from_parent<
		bc::traits::max(0,Parent::tensor_dim-1), Parent>;

	using slice_t = std::conditional_t<
		Parent::tensor_dim == 1, scalar_type, slice_type>;

	return slice_t(
		parent.get_stream(),
		parent.get_allocator(),
		parent.get_shape(),
		parent.data() + parent.leading_dim() * index);
}

template<class Parent>
static auto make_ranged_slice(Parent& parent, bc::size_t from, bc::size_t to)
{
	using slice_type = slice_type_from_parent<Parent::tensor_dim, Parent>;
	bc::size_t range = to - from;
	bc::size_t index = parent.leading_dim() * from;
	static_assert(Parent::tensor_dim > 0);
	bc::Dim<Parent::tensor_dim> shape = parent.inner_shape();
	shape[Parent::tensor_dim-1] = range;

	return slice_type(
		parent.get_stream(),
		parent.get_allocator(),
		bc::Shape<Parent::tensor_dim>(shape),
		parent.data() + index);
}

template<class Parent, class ShapeLike>
static auto make_view(Parent& parent, ShapeLike shape) {
	using slice_type = slice_type_from_parent<ShapeLike::tensor_dim, Parent>;

	return slice_type(
		parent.get_stream(),
		parent.get_allocator(),
		bc::Shape<ShapeLike::tensor_dim>(shape),
		parent.data());
}

template<class Parent>
static auto make_scalar(Parent& parent, bc::size_t index) {
	using slice_type = slice_type_from_parent<0, Parent>;

	return slice_type(
		parent.get_stream(),
		parent.get_allocator(),
		bc::Shape<0>(),
		parent.data() + index);
}

template<
	class Parent,
	class ShapeLike,
	class = std::enable_if_t<Parent::tensor_dim != 1>>
auto make_chunk(
		Parent& parent,
		bc::Dim<Parent::tensor_dim> index_points,
		ShapeLike shape)
{
	using slice_type = slice_type_from_parent<
		ShapeLike::tensor_dim, Parent, noncontinuous_memory_tag>;

	return slice_type(
		parent.get_stream(),
		parent.get_allocator(),
		Shape<ShapeLike::tensor_dim>(shape, parent.get_shape()),
		parent.data() + parent.dims_to_index(index_points));
}

template<
	class Parent,
	class ShapeLike,
	class = std::enable_if_t<Parent::tensor_dim == 1>>
auto make_chunk(
		Parent& parent,
		bc::Dim<1> index_points,
		ShapeLike shape)
{
	return make_ranged_slice(
		parent,
		index_points[0],
		index_points[0] + shape[0]);
}



} //ns BC
} //ns exprs
} //ns tensors



#endif /* Array_Slice_H_ */
