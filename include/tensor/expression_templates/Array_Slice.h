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
namespace array_view {

template<class ValueType, class SystemTag>
struct Scalar : Array_Base<Scalar<ValueType, SystemTag>, 0>, Shape<0> {

    using value_type = ValueType;
    using system_tag = SystemTag;
    using pointer_t  = value_type*;
    using shape_t = Shape<0>;

    static constexpr int ITERATOR = 0;
    static constexpr int DIMS = 0;

    pointer_t array;

    template<class Parent>
    BCINLINE Scalar(Parent, pointer_t memptr_)
    : array(memptr_) {}

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

    BCINLINE BC::meta::apply_const_t<pointer_t> memptr() const { return array; }
    BCINLINE       pointer_t memptr()       { return array; }
    BCINLINE const Shape<0>& get_shape() const { return static_cast<const Shape<0>&>(*this); }

};



template<class Parent, int Dimensions, bool Continuous=true>
struct Slice :
		Array_Base<Slice<Parent, Dimensions, Continuous>, Dimensions>,
		std::conditional_t<Continuous || Dimensions==1, Shape<Dimensions>, SubShape<Dimensions>>
	{

	using value_type  = typename Parent::value_type;
	using pointer_t   = decltype(std::declval<Parent>().memptr());
	using system_tag  = typename Parent::system_tag;
	using shape_t = std::conditional_t<Continuous || Dimensions==1, Shape<Dimensions>, SubShape<Dimensions>>;

	static constexpr int ITERATOR =  (Parent::ITERATOR > 1 || !Continuous) ? Dimensions : 1;
	static constexpr int DIMS 	  = Dimensions;

	pointer_t m_array;

	BCINLINE Slice(Parent& parent_, BC::size_t index)
	: shape_t(parent_.get_shape()),
	  m_array(parent_.memptr() + index) {
	}

	BCINLINE Slice(Parent& parent_, const shape_t& shape_, BC::size_t index)
	: shape_t(shape_),
	  m_array(parent_.memptr() + index) {
	}

	BCINLINE const pointer_t memptr() const {
		return m_array;
	}

	BCINLINE pointer_t memptr() {
		return m_array;
	}
    BCINLINE const shape_t& get_shape() const { return static_cast<const shape_t&>(*this); }
};


template<class Parent>
struct Strided_Vector : Array_Base<Strided_Vector<Parent>, 1>, Shape<1> {

    static_assert(Parent::DIMS == 2, "A ROW VIEW MAY ONLY BE CONSTRUCTED FROM A MATRIX");

	using value_type = typename Parent::value_type;
    using allocator_t = typename Parent::allocator_t;
    using system_tag = typename Parent::system_tag;
    static constexpr int ITERATOR = 1;
    static constexpr int DIMS = 1;

    value_type* array_slice;

    BCINLINE Strided_Vector(Parent, value_type* array_slice_, BC::size_t length, BC::size_t stride)
     : Shape<1>(length, stride),
       array_slice(array_slice_) {}

    BCINLINE const auto& operator [] (int i) const {
    	return array_slice[this->leading_dimension(0) * i]; }
    BCINLINE       auto& operator [] (int i)       {
    	return array_slice[this->leading_dimension(0) * i]; }

    template<class... seq> BCINLINE
    const auto& operator () (int i, seq... indexes) const { return *this[i]; }

    template<class... seq> BCINLINE
    auto& operator () (int i, seq... indexes) { return *this[i]; }

    BCINLINE const value_type* memptr() const { return array_slice; }
    BCINLINE       value_type* memptr()       { return array_slice; }

};
}

template<class Parent>
auto make_row(Parent parent, BC::size_t  index) {
    return Array_Slice<Parent, array_view::Strided_Vector<Parent>>(parent, &parent.memptr()[index], parent.dimension(0), parent.leading_dimension(0));
}

template<class Parent>
auto make_diagnol(Parent parent, BC::size_t diagnol_index) {
    BC::size_t stride = parent.leading_dimension(0) + 1;
    BC::size_t length = BC::meta::min(parent.rows(), parent.cols() - diagnol_index);
    BC::size_t ptr_index = diagnol_index > 0 ? parent.leading_dimension(0) * diagnol_index : std::abs(diagnol_index);
    return Array_Slice<Parent, array_view::Strided_Vector<Parent>>(parent, &parent[ptr_index], length, stride);
}




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
	: ViewType(parent_, args_...),
	  m_allocator(parent_.get_allocator()),
	  m_context(parent_.get_context()) {
	}

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
static auto make_slice(Parent& internal, BC::size_t index) {
	return Array_Slice<Parent, array_view::Slice<Parent, Parent::DIMS-1>>(
			internal, internal.slice_ptr_index(index));
}
template<class Parent>
static auto make_ranged_slice(Parent& internal, BC::size_t from, BC::size_t to) {
	constexpr BC::size_t dim_id = Parent::DIMS;
	BC::size_t range = to - from;
	BC::size_t index = internal.slice_ptr_index(from);

	BC::array<dim_id, BC::size_t> inner_shape = internal.inner_shape();

	inner_shape[dim_id-1] = range;
	BC::exprs::Shape<dim_id> new_shape(inner_shape);

	return Array_Slice<Parent, array_view::Slice<Parent, Parent::DIMS>>(
			internal, new_shape, index);
}

template<class Parent, int ndims>
static auto make_view(Parent& parent, BC::array<ndims, BC::size_t> shape) {
	return Array_Slice<Parent, array_view::Slice<Parent, ndims>>(parent, shape, 0);
}

template<class Parent>
	static auto make_scalar(Parent& parent, BC::size_t index) {
		using value_type = BC::meta::propagate_const_t<Parent, typename Parent::value_type>;
		using system_tag = typename Parent::system_tag;

		return Array_Slice<Parent, array_view::Scalar<value_type, system_tag>>(parent, &parent[index]);
	}

template<class Parent, int ndims>
auto make_chunk(Parent& parent, BC::array<Parent::DIMS, int> index_points, BC::array<ndims, int> shape) {
	static_assert(ndims > 1, "TENSOR CHUNKS MUST HAVE DIMENSIONS GREATER THAN 1, USE SCALAR OR RANGED_SLICE OTHERWISE");
	BC::size_t index = parent.dims_to_index(index_points);

	SubShape<ndims> chunk_shape = SubShape<ndims>(shape, parent.get_shape());
	return Array_Slice<Parent,  array_view::Slice<Parent, ndims, false>>(parent, chunk_shape, index);
}


}
}


#endif /* Array_Slice_H_ */
