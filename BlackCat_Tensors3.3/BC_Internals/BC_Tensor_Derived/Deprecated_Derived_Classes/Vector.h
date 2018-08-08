///*
// * Vector.h
// *
// *  Created on: Dec 30, 2017
// *      Author: joseph
// */
//
//#ifndef VECTOR_H_
//#define VECTOR_H_
//
//#include "BC_Tensor_Base/Tensor_Base.h"
//
//namespace BC {
//
//template<class T, class Mathlib>
//class Vector : public Tensor_Base<Vector<T, Mathlib>> {
//
//	using parent_class = Tensor_Base<Vector<T, Mathlib>>;
//
//public:
//
//	__BCinline__ static constexpr int DIMS() { return 1; }
//	using parent_class::operator=;
//	using parent_class::operator[];
//	using parent_class::operator();
//	//constructors---------------------------------------------------
//
//	Vector(const Vector&  t) : parent_class(t) {}
//	Vector(		 Vector&& t) : parent_class(std::move(t)) {}
//
//	explicit Vector(int dim = 0) : parent_class(Shape<1>(dim)) {}
//	explicit Vector(Shape<DIMS()> shape) : parent_class(shape) {}
//
//	template<class... params>
//	Vector(const params&... p) : parent_class(p...) {}
//
//	//copy_operators--------------------------------------------------
//
//	Vector& operator =(const Vector&  t) { return parent_class::operator=(t); }
//	Vector& operator =(	     Vector&& t) { return parent_class::operator=(std::move(t)); }
//
//	template<class U>
//	Vector& operator = (const Vector<U, Mathlib>& t) { return parent_class::operator=(t); }
//};
//}
//
//#endif /* VECTOR_H_ */
