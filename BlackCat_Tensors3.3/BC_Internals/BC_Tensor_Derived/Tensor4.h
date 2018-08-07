///*
// * Tensor.h
// *
// *  Created on: Dec 30, 2017
// *      Author: joseph
// */
//
//#ifndef BC_TENSOR_N4H
//#define BC_TENSOR_N4H
//
//#include "BC_Tensor_Base/Tensor_Base.h"
//
//namespace BC {
//
//template<class T, class Mathlib>
//class Tensor4 : public Tensor_Base<Tensor4<T, Mathlib>> {
//
//	using parent_class = Tensor_Base<Tensor4<T, Mathlib>>;
//
//public:
//
//	__BCinline__ static constexpr int DIMS() { return 4; }
//	using parent_class::operator=;
//	using parent_class::operator[];
//	using parent_class::operator();
//
//	//constructors---------------------------------------------------
//
//	Tensor4(const Tensor4&  v) : parent_class(v) {}
//	Tensor4(	  Tensor4&& v) : parent_class(v) {}
//
//	explicit Tensor4(int a = 0,int b = 1,int c = 1, int d = 1) : parent_class(Shape<4>(a,b,c,d)) {}
//	explicit Tensor4(Shape<DIMS()> shape) : parent_class(shape)  {}
//
//	template<class... params>
//	Tensor4(const params&... p) : parent_class(p...) {}
//
//	//copy_operators--------------------------------------------------
//
//	Tensor4& operator =(const Tensor4& t)  { return parent_class::operator=(t); }
//	Tensor4& operator =(	  Tensor4&& t) { return parent_class::operator=(t); }
//
//	template<class U>
//	Tensor4& operator = (const Tensor4<U, Mathlib>& t) { return parent_class::operator=(t); }
//
//};
//} //End Namespace BC
//
//#endif /* MATRIX_H */
