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
#include "Implementation_Core/Tensor_Core.cu"

namespace BC {

using MTF::ifte;		//if then else
using MTF::isPrim;
using MTF::shell_of;
using MTF::isCore;

template<class T, class derived, class Mathlib, class R>
class TensorBase :
				public Tensor_Operations <T, ifte<isPrim<T>, Tensor_Core<T, Mathlib, R>, T>, derived, Mathlib>,
				public Tensor_Utility    <T, derived, Mathlib, isPrim<T> || isCore<T>::conditional>

{

protected:
	using math_parent  = Tensor_Operations<T, ifte<isPrim<T>, Tensor_Core<T, Mathlib, R>, T>, derived, Mathlib>;
	using functor_type =  ifte<isPrim<T>, Tensor_Core<T, Mathlib, R>, T>;
	using child = typename Tensor_Core<T, Mathlib, R>::child;
	template<class> struct DISABLED;
	static constexpr bool GENUINE_TENSOR = isPrim<T> || isCore<T>::conditional;
	functor_type black_cat_array;

public:
	using math_parent::operator=;
	operator  const derived&() const { return static_cast<const derived&>(*this); }
	operator  		derived&() 		 { return static_cast<	    derived&>(*this); }


	template<class... params> TensorBase(const params&... p) : black_cat_array(p...) { }

	template<class    U> using deriv   = typename shell_of<derived>::type<U, Mathlib>;
	template<class... U> using functor = typename shell_of<functor_type>::type<U...>;

	template<class var>
	static std::vector<int> shapeOf(const var& v) {
		std::vector<int> sh(v.rank());
		for (int i = 0; i < v.rank(); ++i){
			sh[i] = v.dimension(i);
		}
		return sh;
	}
	template<class U>
	TensorBase(const deriv<U>& tensor) : black_cat_array(shapeOf(tensor)) {
		this->assert_same_size(tensor);
		Mathlib::copy(this->black_cat_array, tensor.data(), this->size());
	}
	template<class... U> TensorBase(const deriv<functor<U...>>& tensor) : black_cat_array(tensor.black_cat_array) {}

	TensorBase(		 derived&& tensor) : black_cat_array(tensor.black_cat_array){}
	TensorBase(const derived&  tensor) : black_cat_array(tensor.black_cat_array){}

	int rank() const { return black_cat_array.rank(); }
	int size() const { return black_cat_array.size(); }
	int rows() const { return black_cat_array.rows(); }
	int cols() const { return black_cat_array.cols(); }
	int LD_rows() const { return black_cat_array.LD_rows(); }
	int LD_cols() const { return black_cat_array.LD_cols(); }
	void resetShape(std::vector<int> sh) { black_cat_array.resetShape(sh); }

	int dimension(int i)		const { return black_cat_array.dimension(i); }
	void printDimensions() 		const { black_cat_array.printDimensions();   }
	void printLDDimensions()	const { black_cat_array.printLDDimensions(); }

	const auto InnerShape() const 			{ return black_cat_array.InnerShape(); }
	const auto OuterShape() const 			{ return black_cat_array.OuterShape(); }
	const functor_type& _data() const { return black_cat_array; }
		  functor_type& _data()		  { return black_cat_array; }

	derived& operator =(const derived& tensor) {
		this->assert_same_size(tensor);
		Mathlib::copy(this->data(), tensor.data(), this->size());
		return *this;
	}
};

}


#endif /* TENSOR_BASE_H_ */

