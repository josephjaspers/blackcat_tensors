
/*
 * Tensor_Base.h
 *
 *  Created on: Jan 6, 2018
 *      Author: joseph
 */

#ifndef TENSOR_BASE_H_
#define TENSOR_BASE_H_

#include "Implementation_Core/Tensor_Operations.h"
#include "Implementation_Core/Tensor_Utility.h"
#include "Implementation_Core/Tensor_Core.h"
#include "Implementation_Core/Tensor_Initializer.h"
#include "Implementation_Core/Determiners.h"

namespace BC {

template<class derived>
class TensorBase :
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

	static constexpr int DIMS 		= ranker<derived>::value;

public:
	template<class>
	friend class TensorBase;

	using math_parent::operator=;

	template<class... params>
	explicit TensorBase(const params&... p) : initializer(p...) {}

	operator const derived& () const { return static_cast<const derived&>(*this); }
	operator	   derived& () 		 { return static_cast< derived&>(*this); }

	derived& operator =(derived&& tensor) {
		this->assert_same_size(tensor);
		std::swap(this->black_cat_array, tensor.black_cat_array);
		return this;
	}

	template<class T>
	std::enable_if_t<MTF::isPrimitive<T>::conditional, derived&> operator = (TensorBase<T>&& tensor) {
		//Only enabled for Tensor_Core types
		this->assert_same_size(tensor);
		std::swap(this->black_cat_array.array, tensor.black_cat_array.array);
	}

	derived& operator =(const derived& tensor) {
		this->assert_same_size(tensor);
		Mathlib::copy(this->data(), tensor.data(), this->size());
		return *this;
	}
	derived& operator =(_scalar<derived> scalar) {
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


	__BCinline__ const functor_type& _data() const { return this->black_cat_array; }
	__BCinline__ 	   functor_type& _data()		  { return this->black_cat_array; }

private:
	const auto slice(int i) const { return this->black_cat_array.slice(i); }
		  auto slice(int i) 	  { return this->black_cat_array.slice(i); }

	const auto scalar(int i) const { return this->black_cat_array.scalar(i); }
		  auto scalar(int i)	   { return this->black_cat_array.scalar(i); }

	const auto row_(int i) const { return this->black_cat_array.row(i); }
		  auto row_(int i)	     { return this->black_cat_array.row(i); }

public:
		  auto operator [] (int i) 		 { return getSlice(i); }
	const auto operator [] (int i) const { return getSlice(i); }

	const auto getScalar(int i) const { return base<0>::type<Tensor_Scalar<functor_type>, Mathlib>(scalar(i)); }
		  auto getScalar(int i) { return base<0>::type<Tensor_Scalar<functor_type>, Mathlib>(scalar(i)); }

	const auto getSlice(int i) const {
		static_assert(derived::DIMS() > 0, "SCALAR SLICE IS NOT DEFINED");
		return typename base<DIMS>::template slice<decltype(slice(0)), Mathlib>(slice(i)); }
		  auto getSlice(int i) 		 {
				static_assert(derived::DIMS() > 0, "SCALAR SLICE IS NOT DEFINED");

			  return typename base<DIMS>::template slice<decltype(slice(0)), Mathlib>(slice(i)); }

	const auto row(int i) const {
		static_assert(DIMS == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return typename base<1>::template type<decltype(row_(0)), Mathlib>(row_(i));
	}
	auto row(int i) {
		static_assert(DIMS == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return typename base<1>::template type<decltype(row_(0)), Mathlib>(row_(i));
	}
	const auto col(int i) const {
		static_assert(DIMS == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return (*this)[i];
	}
	auto col(int i) {
		static_assert(DIMS == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return (*this)[i];
	}

	const auto operator() (int i) const { return getScalar(i); }
		  auto operator() (int i) 	    { return getScalar(i); }

	const auto operator() () const { return *this; }
		  auto operator() () 	   { return *this; }

	template<class... integers> const auto operator() (int i, integers... ints) const { return (*this)[i](ints...); }
	template<class... integers> 	  auto operator() (int i, integers... ints) 	  { return (*this)[i](ints...); }

	void reshape(std::vector<int> sh) {
		if (sh.size() > derived::DIMS()){
			throw std::invalid_argument("can only reshape to tensors of dimensionality <= DIMS()");
		}
		if (sh.size () < derived::DIMS()) {
			for (int i = sh.size(); i < derived::DIMS(); ++i) {
				sh.push_back(1);
			}
		}

		this->black_cat_array.resetShape(sh);
	}

//	template<class var>
//	static auto shapeOf(const var& v) {
//		return reference_array<const var&>(v);
//	}
//
//		template<class d> static std::enable_if_t<d::DIMS() <= 1, 	    d&> flatten(	  TensorBase<d>& tensor) { return tensor; }
//		template<class d> static std::enable_if_t<d::DIMS() <= 1, const d&> flatten(const TensorBase<d>& tensor) { return tensor; }
//
//		template<class d> static std::enable_if_t<(d::DIMS() > 1),
//				typename base<1>::template type <_scalar<d>, _mathlib<d>>
//				>  flatten(TensorBase<d>& tensor) {
//
//				 Vector<Tensor_Core<Vector<_scalar<d>, _mathlib<d>>>, _mathlib<d>> flat(std::true_type());
//				 flat.black_cat_array.array = tensor.black_cat_array.array;
//				 flat.black_cat_array.is = &(tensor.black_cat_array.os[d::DIMS() - 1]);
//				 flat.black_cat_array.os = &(tensor.black_cat_array.os[d::DIMS() - 1]);
//
//		}
//
//		template<class d> static std::enable_if_t<(d::DIMS() > 1),
//				const typename base<1>::template type <_scalar<d>, _mathlib<d>>
//				>  flatten(const TensorBase<d>& tensor) {
//
//				 Vector<Tensor_Core<Vector<_scalar<d>, _mathlib<d>>>, _mathlib<d>> flat(std::true_type());
//				 flat.black_cat_array.array = tensor.black_cat_array.array;
//				 flat.black_cat_array.is = &(tensor.black_cat_array.os[d::DIMS() - 1]);
//				 flat.black_cat_array.os = &(tensor.black_cat_array.os[d::DIMS() - 1]);
//
//		}

};

}


#endif /* TENSOR_BASE_H_ */

