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
#include "Implementation_Core/Determiners.h"
namespace BC {

template<class derived>
class TensorBase : public Tensor_Operations <derived>, public Tensor_Utility<derived> {

protected:

	template<class> struct DISABLED;

	using self 			= TensorBase<derived>;
	using math_parent  	= Tensor_Operations<derived>;
	using functor_type 	= _functor<derived>;
	using Mathlib 		= _mathlib<derived>;

	template<class    U> using deriv   = typename shell_of<derived>::type<U, Mathlib>;
	template<class... U> using functor = typename shell_of<functor_type>::type<U...>;

	static constexpr int RANK 		= ranker<derived>::type::inner_rank;
	static constexpr int LD_RANK 	= ranker<derived>::type::outer_rank;
	static constexpr bool GENUINE_TENSOR =isPrim<_fscal<derived>> || isCore<_functor<derived>>::conditional;

	functor_type black_cat_array;

public:
	using math_parent::operator=;

	operator  const derived&() const { return static_cast<const derived&>(*this); }
	operator  		derived&() 		 { return static_cast<	    derived&>(*this); }

	TensorBase(		 derived&& tensor) : black_cat_array(tensor.black_cat_array){}
	TensorBase(const derived&  tensor) : black_cat_array(tensor.black_cat_array){}

	template<class... U> TensorBase(const deriv<functor<U...>>& tensor)
			: black_cat_array(tensor.black_cat_array) {}
	template<class atLeastOneParam, class... params> TensorBase(const atLeastOneParam& alop, const params&... p)
			: black_cat_array(alop, p...) { }

	template<class U>
	TensorBase(const deriv<U>& tensor) : black_cat_array(shapeOf(tensor)) {
		this->assert_same_size(tensor);
		Mathlib::copy(this->black_cat_array, tensor.data(), this->size());
	}

	derived& operator =(const derived& tensor) {
		this->assert_same_size(tensor);
		Mathlib::copy(this->data(), tensor.data(), this->size());
		return *this;
	}

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

	const auto slice(int i) const { return black_cat_array.slice(i); }
		  auto slice(int i) 	  { return black_cat_array.slice(i); }

		  auto operator [] (int i) 		 { return typename base<RANK>::slice<decltype(slice(0)), Mathlib>(slice(i)); }
	const auto operator [] (int i) const { return typename base<RANK>::slice<decltype(slice(0)), Mathlib>(slice(i)); }

	const auto& operator() (int i) const { return this->data()[i]; }
		  auto& operator() (int i) 	     { return this->data()[i]; }

	template<class var>
	static std::vector<int> shapeOf(const var& v) {
		std::vector<int> sh(v.rank());
		for (int i = 0; i < v.rank(); ++i) {
			sh[i] = v.dimension(i);
		}
		return sh;
	}
};

}


#endif /* TENSOR_BASE_H_ */

