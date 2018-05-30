/*
 * Tensor.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_N_H
#define BC_TENSOR_N_H

#include "BC_Tensor_Base/Tensor_Base.h"

namespace BC {


template<class T, class Mathlib>
class Tensor4 : public Tensor_Base<Tensor4<T, Mathlib>> {

	using parent_class = Tensor_Base<Tensor4<T, Mathlib>>;

public:
	using parent_class::operator=;
	using parent_class::operator[];
	using parent_class::operator();

	__BCinline__ static constexpr int DIMS() { return 4; }

	Tensor4(const Tensor4&  v) : parent_class(v) {}
	Tensor4(	  Tensor4&& v) : parent_class(v) {}
	Tensor4(const Tensor4&& v) : parent_class(v) {}
	explicit Tensor4(int a = 0,int b = 1,int c = 1, int d = 1) : parent_class(Shape<4>(a,b,c,d)) {}
	explicit Tensor4(Shape<DIMS()> shape) : parent_class(shape)  {}

	template<class U> 		  Tensor4(const Tensor4<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> 		  Tensor4(	    Tensor4<U, Mathlib>&& t) : parent_class(t) {}

	Tensor4& operator =(const Tensor4& t)  { return parent_class::operator=(t); }
	Tensor4& operator =(const Tensor4&& t) { return parent_class::operator=(t); }
	Tensor4& operator =(	  Tensor4&& t) { return parent_class::operator=(t); }
	template<class U>
	Tensor4& operator = (const Tensor4<U, Mathlib>& t) { return parent_class::operator=(t); }

private:

	template<class U> friend class Tensor_Base;
	template<class U> friend class Tensor_Operations;
	template<class... params> Tensor4(const params&... p) : parent_class(p...) {}


};

} //End Namespace BC

#endif /* MATRIX_H */
