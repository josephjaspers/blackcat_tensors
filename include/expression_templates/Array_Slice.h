/*
 * Array_Slice.h
 *
 *  Created on: Dec 24, 2018
 *      Author: joseph
 */

#ifndef Array_Slice_H_
#define Array_Slice_H_

#include "Internal_Common.h"
#include "Array_Base.h"
#include "Shape.h"

namespace BC {
namespace et {

//Floored decrement just returns the max(param - 1, 0)

template<class PARENT, int ndimensions, bool continuous=true>
struct Array_Slice :
		Array_Base<Array_Slice<PARENT, ndimensions, continuous>, ndimensions>,
		std::conditional_t<continuous, Shape<ndimensions>, SubShape<ndimensions>>
	{

	using scalar_t = typename PARENT::scalar_t;
	using pointer_t = decltype(std::declval<PARENT>().memptr());
	using allocator_t = typename PARENT::allocator_t;
	using system_tag = typename PARENT::system_tag;
	using shape_t = std::conditional_t<continuous, Shape<ndimensions>, SubShape<ndimensions>>;

	__BCinline__
	static constexpr BC::size_t ITERATOR() {
		return (PARENT::ITERATOR() > 1 or !continuous)
				? ndimensions : 1;
	}
	__BCinline__
	static constexpr BC::size_t DIMS() {
		return ndimensions;
	}

	pointer_t m_array;

	__BCinline__
	Array_Slice(PARENT parent_, BC::size_t index)
	: shape_t(parent_.as_shape()),
	  m_array(parent_.memptr() + index) {
	}
	__BCinline__
	Array_Slice(PARENT parent_, const shape_t& shape_, BC::size_t index)
	: shape_t(shape_),
	  m_array(parent_.memptr() + index) {
	}


	__BCinline__
	const scalar_t* memptr() const {
		return m_array;
	}
	__BCinline__
	scalar_t* memptr() {
		return m_array;
	}

};

	template<class parent_t>
	static auto make_slice(parent_t internal, BC::size_t index) {
		return Array_Slice<parent_t, parent_t::DIMS()-1>(internal, internal.slice_ptr_index(index));
	}
	template<class parent_t>
	static auto make_ranged_slice(parent_t internal, BC::size_t from, BC::size_t to) {
		constexpr BC::size_t dim_id = parent_t::DIMS();
		BC::size_t range = to - from;
		BC::size_t index = internal.slice_ptr_index(from);

		BC::array<dim_id, BC::size_t> inner_shape = internal.inner_shape();

		inner_shape[dim_id-1] = range;
		BC::et::Shape<dim_id> new_shape(inner_shape);

		return Array_Slice<parent_t, parent_t::DIMS()>(
				internal, new_shape, index);
	}

	template<class parent_t, BC::dim_t ndims>
	static auto make_view(parent_t parent, BC::array<ndims, BC::size_t> shape) {
		return Array_Slice<parent_t, ndims>(parent, shape, 0);
	}

	template<class parent_t, BC::dim_t ndims>
	auto make_chunk(parent_t parent, BC::array<parent_t::DIMS(), int> index_points, BC::array<ndims, int> shape) {
		BC::size_t index = parent.dims_to_index(index_points);
		SubShape<ndims> chunk_shape = SubShape<ndims>(shape, parent.as_shape());
		return Array_Slice<parent_t, ndims, false>(parent, chunk_shape, index);
	}

}
}

#endif /* Array_Slice_H_ */
