/*
 * Tensor.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_N5_H
#define BC_TENSOR_N5_H

#include "BC_Tensor_Base/Tensor_Base.h"

namespace BC {

template<class T, class Mathlib>
class Tensor5 : public Tensor_Base<Tensor5<T, Mathlib>> {

	using parent_class = Tensor_Base<Tensor5<T, Mathlib>>;

public:
	using parent_class::operator=;
	using parent_class::operator[];
	using parent_class::operator();

	__BCinline__ static constexpr int DIMS() { return 5; }

	Tensor5(const Tensor5&  v) : parent_class(v) {}
	Tensor5(	  Tensor5&& v) : parent_class(v) {}
	Tensor5(const Tensor5&& v) : parent_class(v) {}
	explicit Tensor5(int a = 0,int b = 1,int c = 1, int d = 1, int e = 1) : parent_class(Shape<5>(a,b,c,d,e)) {}
	explicit Tensor5(Shape<DIMS()> shape) : parent_class(shape)  {}

	template<class U> 		  Tensor5(const Tensor5<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> 		  Tensor5(	    Tensor5<U, Mathlib>&& t) : parent_class(t) {}

	Tensor5& operator =(const Tensor5& t)  { return parent_class::operator=(t); }
	Tensor5& operator =(const Tensor5&& t) { return parent_class::operator=(t); }
	Tensor5& operator =(	  Tensor5&& t) { return parent_class::operator=(t); }
	template<class U>
	Tensor5& operator = (const Tensor5<U, Mathlib>& t) { return parent_class::operator=(t); }

private:

	template<class> friend class Tensor_Base;
	template<class> friend class Tensor_Operations;
	template<class... params> Tensor5(const params&... p) : parent_class(p...) {}

};

} //End Namespace BC

#endif /* MATRIX_H */
