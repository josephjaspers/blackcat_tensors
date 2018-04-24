
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

namespace BC {

struct TensorType {};

template<class derived>
class TensorBase :
		public TensorType,
		public Tensor_Operations <derived>,
		public Tensor_Utility<derived>,
		public TensorInitializer<derived>
{

protected:

	using self 			= TensorBase<derived>;
	using math_parent  	= Tensor_Operations<derived>;
	using initializer 	= TensorInitializer<derived>;
	using functor_type 	= _functor<derived>;
	using scalar_type	= _scalar<derived>;
	using Mathlib 		= _mathlib<derived>;
	struct DISABLED;

	static constexpr int DIMS = ranker<derived>::value;
	static constexpr int CONTINUOUS() { return functor_type::CONTINUOUS(); }
	template<class> friend class TensorBase;

public:
	using math_parent::operator=;

	operator const derived& () const { return static_cast<const derived&>(*this); }
	operator	   derived& () 		 { return static_cast< 		derived&>(*this); }


	template<class... params> explicit TensorBase(const params&... p) : initializer(p...) {}

	//move only defined for Tensor_Core's
	using move_parameter = std::conditional_t<pCore_b<functor_type>, derived&&, DISABLED>;
	TensorBase(		 move_parameter tensor) : initializer(std::move(tensor)) {}
	TensorBase(const TensorBase& 	tensor) : initializer(tensor) {}


	derived& operator =(move_parameter tensor) {
		auto tmp = this->black_cat_array;
		this->black_cat_array = tensor.black_cat_array;
		tensor.black_cat_array = tmp;

		return *this;
	}

	derived& operator =(const derived& tensor) {
		this->assert_same_size(tensor);
		Mathlib::copy(this->data(), tensor.data(), this->size());
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
	int LD_rows() const { return this->black_cat_array.LD_rows(); }
	int LD_cols() const { return this->black_cat_array.LD_cols(); }

	int dimension(int i)		const { return this->black_cat_array.dimension(i); }
	void printDimensions() 		const { this->black_cat_array.printDimensions();   }
	void printLDDimensions()	const { this->black_cat_array.printLDDimensions(); }

	const auto innerShape() const 			{ return this->black_cat_array.innerShape(); }
	const auto outerShape() const 			{ return this->black_cat_array.outerShape(); }


	 const functor_type& _data() const { return this->black_cat_array; }
	 	   functor_type& _data()		  { return this->black_cat_array; }

private:
	const auto slice_impl(int i) const { return this->black_cat_array.slice(i); }
		  auto slice_impl(int i) 	  { return this->black_cat_array.slice(i); }

	const auto scalar_impl(int i) const { return this->black_cat_array.scalar(i); }
		  auto scalar_impl(int i)	   { return this->black_cat_array.scalar(i); }

	const auto row_impl(int i) const { return this->black_cat_array.row(i); }
		  auto row_impl(int i)	     { return this->black_cat_array.row(i); }
public:
	const auto operator [] (int i) const { return slice(i); }
		  auto operator [] (int i) 		 { return slice(i); }

	const auto scalar(int i) const { return base<0>::type<Tensor_Scalar<functor_type>, Mathlib>(scalar_impl(i)); }
		  auto scalar(int i) 	   { return base<0>::type<Tensor_Scalar<functor_type>, Mathlib>(scalar_impl(i)); }

	const auto slice(int i) const {
		static_assert(derived::DIMS() > 0, "SCALAR SLICE IS NOT DEFINED");
		return typename base<DIMS>::template slice<decltype(slice_impl(0)), Mathlib>(slice_impl(i)); }

		auto slice(int i) 		  {
		static_assert(derived::DIMS() > 0, "SCALAR SLICE IS NOT DEFINED");
		return typename base<DIMS>::template slice<decltype(slice_impl(0)), Mathlib>(slice_impl(i)); }

	const auto row(int i) const {
		static_assert(DIMS == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return typename base<1>::template type<decltype(row_impl(0)), Mathlib>(row_impl(i));
	}
		  auto row(int i) 		{
		static_assert(DIMS == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return typename base<1>::template type<decltype(row_impl(0)), Mathlib>(row_impl(i));
	}
	const auto col(int i) const {
		static_assert(DIMS == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return (*this)[i];
	}
		 auto col(int i) {
		static_assert(DIMS == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
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
	void resetShape(integers... ints) {
		static_assert(MTF::is_integer_sequence<integers...>, "MUST BE INTEGER LIST");
		this->black_cat_array.resetShape(ints...);
	}
	template<class... integers>
	auto reshape(integers... ints) {
		static_assert(MTF::is_integer_sequence<integers...>, "MUST BE INTEGER LIST");
		using type = typename base<sizeof...(integers)>::template type<Tensor_Reshape<functor_type, sizeof...(integers)>, Mathlib>;
		return type(this->black_cat_array.reshape(ints...));

	}

};

}


#endif /* TENSOR_BASE_H_ */

