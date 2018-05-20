/*
 * Tensor.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_N4_H
#define BC_TENSOR_N4_H

#include "BC_Tensor_Base/Tensor_Base.h"

namespace BC {


template<int dimensions>
struct dimension {

template<class T, class Mathlib>
class Tensor : public Tensor_Base<Tensor<T, Mathlib>> {

	using parent_class = Tensor_Base<Tensor<T, Mathlib>>;

public:
	using parent_class::operator=;
	using parent_class::operator[];
	using parent_class::operator();

	__BCinline__ static constexpr int DIMS() { return dimensions; }

	Tensor(const Tensor&  v) : parent_class(v) {}
	Tensor(	  Tensor&& v) : parent_class(v) {}
	Tensor(const Tensor&& v) : parent_class(v) {}

	template<class... integers>
	explicit Tensor(int x = 0, integers... ints) : parent_class(array(x, ints...)) {}

	template<class U> 		  Tensor(const Tensor<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> 		  Tensor(	    Tensor<U, Mathlib>&& t) : parent_class(t) {}

	Tensor& operator =(const Tensor& t)  { return parent_class::operator=(t); }
	Tensor& operator =(const Tensor&& t) { return parent_class::operator=(t); }
	Tensor& operator =(	  Tensor&& t) { return parent_class::operator=(t); }
	template<class U>
	Tensor& operator = (const Tensor<U, Mathlib>& t) { return parent_class::operator=(t); }

private:

	template<class U> friend class Tensor_Base;
	template<class U> friend class Tensor_Operations;
	template<class... params> Tensor(const params&... p) : parent_class(p...) {}
};

};

} //End Namespace BC

#endif /* MATRIX_H */
