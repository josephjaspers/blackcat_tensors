
/*
 * Tensor_Base.h
 *
 *  Created on: Jan 6, 2018
 *      Author: joseph
 */

#ifndef TENSOR_BASE_H_
#define TENSOR_BASE_H_

#include "Tensor_Operations.h"
#include "Tensor_Utility.h"
#include "Tensor_Initializer.h"
#include "Tensor_Shaping.h"

namespace BC {


template<class derived>
class Tensor_Base :
		public Tensor_Operations<derived>,
		public Tensor_Utility<derived>,
		public Tensor_Initializer<derived>,
		public Tensor_Shaping<derived>
{

protected:

	using self 			= Tensor_Base<derived>;
	using operations  	= Tensor_Operations<derived>;
	using initializer 	= Tensor_Initializer<derived>;
	using utility		= Tensor_Utility<derived>;

	using functor_type 	= _functor<derived>;
	using scalar_type	= _scalar<derived>;
	using mathlib_type 	= _mathlib<derived>;

	static constexpr int DIMS() { return dimension_of<derived>::value; }
	static constexpr int ITERATOR() { return functor_type::ITERATOR(); }

	template<class> friend class Tensor_Base;

public:
	using operations::operator=;

	operator const derived& () const { return static_cast<const derived&>(*this); }
	operator	   derived& () 		 { return static_cast< 		derived&>(*this); }


	template<class... params> explicit Tensor_Base(const params&... p) : initializer(p...) {}

	//move only defined for primary cores (this is to ensure slices/chunks/reshapes apply copies)
	using move_parameter = std::conditional_t<pCore_b<functor_type>, derived&&, DISABLED>;
	Tensor_Base(move_parameter tensor) : initializer(std::move(tensor)) {}
	Tensor_Base(const Tensor_Base& 	tensor) : initializer(tensor) {}


	derived& operator =(move_parameter tensor) {
		auto tmp = this->black_cat_array;
		this->black_cat_array = tensor.black_cat_array;
		tensor.black_cat_array = tmp;

		return *this;
	}

	derived& operator =(const derived& tensor) {
		this->assert_same_size(tensor);
		mathlib_type::copy(this->data(), tensor.data(), this->size());
		return *this;
	}
	derived& operator =(scalar_type scalar) {
		this->fill(scalar);
		return *this;
	}

	int dims() const { return this->black_cat_array.dims(); }
	int size() const { return this->black_cat_array.size(); }
	int rows() const { return this->black_cat_array.rows(); }
	int cols() const { return this->black_cat_array.cols(); }
	int outer_dimension() const { return this->black_cat_array.outer_dimension(); }

	int ld1() const { return this->black_cat_array.ld1(); }
	int ld2() const { return this->black_cat_array.ld2(); }

	int dimension(int i)		const { return this->black_cat_array.dimension(i); }
	void print_dimensions() 		const { this->black_cat_array.print_dimensions();   }
	void print_outer_dimensions()	const { this->black_cat_array.print_outer_dimensions(); }

	const auto inner_shape() const 			{ return this->black_cat_array.inner_shape(); }
	const auto outer_shape() const 			{ return this->black_cat_array.outer_shape(); }

	 const functor_type& data() const { return this->black_cat_array; }
	 	   functor_type& data()		  { return this->black_cat_array; }

private:
	const auto slice_impl(int i) const { return this->black_cat_array.slice(i); }
		  auto slice_impl(int i) 	  { return this->black_cat_array.slice(i);  }

	const auto scalar_impl(int i) const { return this->black_cat_array.scalar(i); }
		  auto scalar_impl(int i)	   { return this->black_cat_array.scalar(i);  }

	const auto row_impl(int i) const { return this->black_cat_array.row(i); }
		  auto row_impl(int i)	     { return this->black_cat_array.row(i); }
public:
	const auto operator [] (int i) const { return slice(i); }
		  auto operator [] (int i) 		 { return slice(i); }

	const auto scalar(int i) const { return tensor_of<0>::type<Tensor_Scalar<functor_type>, mathlib_type>(scalar_impl(i)); }
		  auto scalar(int i) 	   { return tensor_of<0>::type<Tensor_Scalar<functor_type>, mathlib_type>(scalar_impl(i)); }

	const auto slice(int i) const {
		static_assert(DIMS() > 0, "SCALAR SLICE IS NOT DEFINED");
		return typename tensor_of<DIMS()>::template slice<decltype(slice_impl(0)), mathlib_type>(slice_impl(i)); }

		  auto slice(int i) 	  {
		static_assert(derived::DIMS() > 0, "SCALAR SLICE IS NOT DEFINED");
		return typename tensor_of<DIMS()>::template slice<decltype(slice_impl(0)), mathlib_type>(slice_impl(i)); }

	const auto row(int i) const {
		static_assert(DIMS() == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return typename tensor_of<1>::template type<decltype(row_impl(0)), mathlib_type>(row_impl(i));
	}
		  auto row(int i) 		{
		static_assert(DIMS() == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return typename tensor_of<1>::template type<decltype(row_impl(0)), mathlib_type>(row_impl(i));
	}
	const auto col(int i) const {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return (*this)[i];
	}
		 auto col(int i) {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return (*this)[i];
	}


	const auto operator() (int i) const { return scalar(i); }
		  auto operator() (int i) 	    { return scalar(i); }

	const auto& operator() () const { return *this; }
		  auto& operator() () 	    { return *this; }

	template<class... integers> const auto operator() (int i, integers... ints) const  {
		static_assert(MTF::is_integer_sequence<integers...>, "MUST BE INTEGER LIST");
		return (*this)[i](ints...);
	}

	template<class... integers> 	  auto operator() (int i, integers... ints) {
		static_assert(MTF::is_integer_sequence<integers...>, "MUST BE INTEGER LIST");
		return (*this)[i](ints...);
	}

	template<class... integers>
	void resize(integers... ints) {
		this->black_cat_array.resetShape(ints...);
	}
};

}

#endif /* TENSOR_BASE_H_ */

