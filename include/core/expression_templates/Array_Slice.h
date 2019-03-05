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


template<class Parent>
struct ArrayScalarExpr : Array_Base<ArrayScalarExpr<Parent>, 0>, Shape<0> {

    using value_type = typename Parent::value_type;
	using pointer_t   = decltype(std::declval<Parent>().memptr());
    using allocator_t = typename Parent::allocator_t;
    using system_tag = typename Parent::system_tag;
    using shape_t = Shape<0>;

    static constexpr int ITERATOR = 0;
    static constexpr int DIMS = 0;

    pointer_t array;

    BCINLINE ArrayScalarExpr(Parent parent_, BC::size_t index)
    : array(&(parent_.memptr()[index])) {}

    BCINLINE const auto& operator [] (int index) const { return array[0]; }
    BCINLINE       auto& operator [] (int index)       { return array[0]; }

    template<class... integers> BCINLINE
    auto& operator ()(integers ... ints) {
        return array[0];
    }
    template<class... integers> BCINLINE
    const auto& operator ()(integers ... ints) const {
        return array[0];
    }

    BCINLINE const pointer_t memptr() const { return array; }
    BCINLINE       pointer_t memptr()       { return array; }
    BCINLINE const Shape<0>& get_shape() const { return static_cast<const Shape<0>&>(*this); }

};



template<class Parent, int Dimensions, bool Continuous=true>
struct ArraySliceExpr :
		Array_Base<ArraySliceExpr<Parent, Dimensions, Continuous>, Dimensions>,
		std::conditional_t<Continuous, Shape<Dimensions>, SubShape<Dimensions>>
	{

	using value_type  = typename Parent::value_type;
	using pointer_t   = decltype(std::declval<Parent>().memptr());
	using allocator_t = typename Parent::allocator_t;
	using system_tag  = typename Parent::system_tag;
	using shape_t = std::conditional_t<Continuous, Shape<Dimensions>, SubShape<Dimensions>>;

	static constexpr int ITERATOR =  (Parent::ITERATOR > 1 || !Continuous) ? Dimensions : 1;
	static constexpr int DIMS 	  = Dimensions;

	pointer_t m_array;

	BCINLINE
	ArraySliceExpr(Parent& parent_, BC::size_t index)
	: shape_t(parent_.get_shape()),
	  m_array(parent_.memptr() + index) {
	}

	BCINLINE
	ArraySliceExpr(Parent& parent_, const shape_t& shape_, BC::size_t index)
	: shape_t(shape_),
	  m_array(parent_.memptr() + index) {
	}

	BCINLINE
	const pointer_t memptr() const {
		return m_array;
	}
	BCINLINE
	pointer_t memptr() {
		return m_array;
	}

    BCINLINE const shape_t& get_shape() const { return static_cast<const shape_t&>(*this); }

};


template<class Parent, int Dimensions, bool Continuous=true>
struct Array_Slice :
		std::conditional_t<Dimensions == 0,
		ArrayScalarExpr<Parent>,
		ArraySliceExpr<Parent, Dimensions, Continuous>> {

	using super_t = std::conditional_t<Dimensions == 0,
			ArrayScalarExpr<Parent>,
			ArraySliceExpr<Parent, Dimensions, Continuous>>;

	using shape_t 		 = typename super_t::shape_t;
	using allocator_t 	 = typename Parent::allocator_t;
	using context_t 	 = typename Parent::context_t;
	using full_context_t = decltype(std::declval<Parent>().get_context());

	full_context_t m_context;

	template<class,int, bool> friend class Array_Slice;
	template<int, class, class, class...> friend class Array;

	BCHOT
	Array_Slice(Parent& parent_, BC::size_t index)
	: super_t(parent_, index), m_context(parent_.get_context()) {
	}

	BCHOT
	Array_Slice(Parent& parent_, const shape_t& shape_, BC::size_t index)
	: super_t(parent_, shape_, index), m_context(parent_.get_context()) {
	}

	allocator_t get_allocator() const {
		return BC::allocator_traits<allocator_t>::select_on_container_copy_construction(
			m_context.get_allocator());
	}

	auto& internal_base() { return *this; }
	const auto& internal_base() const { return *this; }

	auto get_context() -> decltype(m_context) { return m_context; }
	auto get_context() const -> decltype(m_context) { return m_context; }
};



	template<class Parent>
	static auto make_slice(Parent& internal, BC::size_t index) {
		return Array_Slice<Parent, Parent::DIMS-1>(internal, internal.slice_ptr_index(index));
	}
	template<class Parent>
	static auto make_ranged_slice(Parent& internal, BC::size_t from, BC::size_t to) {
		constexpr BC::size_t dim_id = Parent::DIMS;
		BC::size_t range = to - from;
		BC::size_t index = internal.slice_ptr_index(from);

		BC::array<dim_id, BC::size_t> inner_shape = internal.inner_shape();

		inner_shape[dim_id-1] = range;
		BC::exprs::Shape<dim_id> new_shape(inner_shape);

		return Array_Slice<Parent, Parent::DIMS>(
				internal, new_shape, index);
	}

	template<class Parent, int ndims>
	static auto make_view(Parent& parent, BC::array<ndims, BC::size_t> shape) {
		return Array_Slice<Parent, ndims>(parent, shape, 0);
	}

	template<class Parent>
		static auto make_scalar(Parent& parent, BC::size_t index) {
			return Array_Slice<Parent, 0, true>(parent, index);
		}

	template<class Parent, int ndims>
	auto make_chunk(Parent& parent, BC::array<Parent::DIMS, int> index_points, BC::array<ndims, int> shape) {
		static_assert(ndims > 1, "TENSOR CHUNKS MUST HAVE DIMENSIONS GREATER THAN 1, USE SCALAR OR RANGED_SLICE OTHERWISE");
		BC::size_t index = parent.dims_to_index(index_points);

		SubShape<ndims> chunk_shape = SubShape<ndims>(shape, parent.get_shape());
		return Array_Slice<Parent, ndims, false>(parent, chunk_shape, index);
	}


}
}


#endif /* Array_Slice_H_ */
