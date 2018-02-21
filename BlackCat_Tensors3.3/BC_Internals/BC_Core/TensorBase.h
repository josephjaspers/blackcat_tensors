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
#include "../BC_MathLibraries/Mathematics_CPU.h"
#include "../BC_MathLibraries/Mathematics_GPU.cu"


namespace BC {

using MTF::ifte;
using MTF::prim;

template<class> struct isCore { static constexpr bool conditional = false; };
template<class a, class b, class c> struct isCore<Tensor_Core<a, b, c>>
{ static constexpr bool conditional = true; };
template<class T>
using isCore_t = isCore<T>;

template<class> struct internal;
template<class t, class ml, template<class,class> class tensor> struct internal<tensor<t, ml>> {
	template<class T, class ML>
	using type = tensor<T, ML>;
};


template<class T, class derived, class Mathlib, class R>
class TensorBase :
				public Tensor_Operations <T, ifte<prim<T>, Tensor_Core<T, Mathlib, R>, T>, derived, Mathlib>,
				public Tensor_Utility    <T, derived, Mathlib, prim<T> | isCore_t<T>::conditional>

{

protected:
	struct DISABLED;
	using accessor = ifte<prim<T>, typename Tensor_Core<T, Mathlib, R>::accessor, DISABLED>;
	using math_parent  = Tensor_Operations<T, ifte<prim<T>, Tensor_Core<T, Mathlib, R>, T>, derived, Mathlib>;
	using functor_type =  ifte<prim<T>, Tensor_Core<T, Mathlib, R>, T>;
	using param_tc = Tensor_Core<T, Mathlib, R>;
	functor_type black_cat_array;

public:

	template<class... params>
	TensorBase(const params&... p) : black_cat_array(p...) {}
	TensorBase(		 derived&& tensor) : black_cat_array(tensor.black_cat_array){}
	TensorBase(const derived&  tensor) : black_cat_array(tensor.black_cat_array){}
	using math_parent::operator=;

	operator  const derived&() const { return static_cast<const derived&>(*this); }
	operator  		derived&() 		 { return static_cast<	    derived&>(*this); }

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

	auto accessor_packet(int index) const { return black_cat_array.accessor_packet(index); }

	const int* InnerShape() const 			{ return black_cat_array.InnerShape(); }
	const int* OuterShape() const 			{ return black_cat_array.OuterShape(); }
	const functor_type& _data() const { return black_cat_array; }
		  functor_type& _data()		  { return black_cat_array; }
};

}


#endif /* TENSOR_BASE_H_ */

