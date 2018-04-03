/*
 * Tensor.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_N_H
#define BC_TENSOR_N_H

#include "BC_Tensor_Base/TensorBase.h"

namespace BC {


template<class T, class Mathlib>
class Tensor4 : public TensorBase<Tensor4<T, Mathlib>> {

	using parent_class = TensorBase<Tensor4<T, Mathlib>>;

public:
	using scalar = T;
	using parent_class::operator=;
	using parent_class::operator[];
	using parent_class::operator();

	__BCinline__ static constexpr int DIMS() { return 4; }

	Tensor4(const Tensor4&  v) : parent_class(v) {}
	Tensor4(	  Tensor4&& v) : parent_class(v) {}
	Tensor4(const Tensor4&& v) : parent_class(v) {}
	explicit Tensor4(int a = 1,int b = 1,int c = 1, int d = 1) : parent_class(std::vector<int>{a,b,c,d}) {}

	template<class U> 		  Tensor4(const Tensor4<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> 		  Tensor4(	    Tensor4<U, Mathlib>&& t) : parent_class(t) {}
	template<class... params> Tensor4(const params&... p) : parent_class(p...) {}

	Tensor4& operator =(const Tensor4& t)  { return parent_class::operator=(t); }
	Tensor4& operator =(const Tensor4&& t) { return parent_class::operator=(t); }
	Tensor4& operator =(	  Tensor4&& t) { return parent_class::operator=(t); }
	template<class U>
	Tensor4& operator = (const Tensor4<U, Mathlib>& t) { return parent_class::operator=(t); }
};

} //End Namespace BC

#endif /* MATRIX_H */
