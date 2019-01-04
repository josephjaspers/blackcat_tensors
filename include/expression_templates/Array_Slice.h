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
namespace et {

template<class,int,bool> class Array_Slice;

template<class Parent, int Dimensions, bool Continuous=true>
struct ArraySliceExpr :
		Array_Base<ArraySliceExpr<Parent, Dimensions, Continuous>, Dimensions>,
		std::conditional_t<Continuous, Shape<Dimensions>, SubShape<Dimensions>>
	{

	using value_type = typename Parent::value_type;
	using pointer_t = decltype(std::declval<Parent>().memptr());
	using allocator_t = typename Parent::allocator_t;
	using system_tag = typename Parent::system_tag;
	using shape_t = std::conditional_t<Continuous, Shape<Dimensions>, SubShape<Dimensions>>;

	static constexpr int ITERATOR =  (Parent::ITERATOR > 1 || !Continuous) ? Dimensions : 1;
	static constexpr int DIMS 	  = Dimensions;

	pointer_t m_array;

	__BCinline__
	ArraySliceExpr(Parent parent_, BC::size_t index)
	: shape_t(parent_.as_shape()),
	  m_array(parent_.memptr() + index) {
	}

	__BCinline__
	ArraySliceExpr(Parent parent_, const shape_t& shape_, BC::size_t index)
	: shape_t(shape_),
	  m_array(parent_.memptr() + index) {
	}

	__BCinline__
	const value_type* memptr() const {
		return m_array;
	}
	__BCinline__
	value_type* memptr() {
		return m_array;
	}

	const auto& get_allocator() const {
		return static_cast<const Array_Slice<Parent, Dimensions, Continuous>&>(*this).m_allocator;
	}

};


template<class Parent, int Dimensions, bool Continuous=true>
struct Array_Slice : ArraySliceExpr<Parent, Dimensions, Continuous> {

	using super_t = ArraySliceExpr<Parent, Dimensions, Continuous>;
	using shape_t = typename super_t::shape_t;
	using allocator_t = typename Parent::allocator_t;

	const allocator_t& m_allocator;

	__BCinline__
	Array_Slice(const Parent& parent_, BC::size_t index)
	: super_t(parent_, index), m_allocator(parent_.get_allocator()) {
	}

	__BCinline__
	Array_Slice(const Parent& parent_, const shape_t& shape_, BC::size_t index)
	: super_t(parent_, shape_, index), m_allocator(parent_.get_allocator()) {
	}
};



	template<class Parent>
	static auto make_slice(const Parent& internal, BC::size_t index) {
		return Array_Slice<Parent, Parent::DIMS-1>(internal, internal.slice_ptr_index(index));
	}
	template<class Parent>
	static auto make_ranged_slice(const Parent& internal, BC::size_t from, BC::size_t to) {
		constexpr BC::size_t dim_id = Parent::DIMS;
		BC::size_t range = to - from;
		BC::size_t index = internal.slice_ptr_index(from);

		BC::array<dim_id, BC::size_t> inner_shape = internal.inner_shape();

		inner_shape[dim_id-1] = range;
		BC::et::Shape<dim_id> new_shape(inner_shape);

		return Array_Slice<Parent, Parent::DIMS>(
				internal, new_shape, index);
	}

	template<class Parent, int ndims>
	static auto make_view(const Parent& parent, BC::array<ndims, BC::size_t> shape) {
		return Array_Slice<Parent, ndims>(parent, shape, 0);
	}

	template<class Parent, int ndims>
	auto make_chunk(const Parent& parent, BC::array<Parent::DIMS, int> index_points, BC::array<ndims, int> shape) {
		static_assert(ndims > 1, "TENSOR CHUNKS MUST HAVE DIMENSIONS GREATER THAN 1, USE SCALAR OR RANGED_SLICE OTHERWISE");
		BC::size_t index = parent.dims_to_index(index_points);

		SubShape<ndims> chunk_shape = SubShape<ndims>(shape, parent.as_shape());
		return Array_Slice<Parent, ndims, false>(parent, chunk_shape, index);
	}


}
}


#endif /* Array_Slice_H_ */
