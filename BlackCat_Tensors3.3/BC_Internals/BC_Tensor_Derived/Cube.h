///*
//
// * Cube.h
// *
// *  Created on: Dec 30, 2017
// *      Author: joseph
// */
//
//#ifndef BC_Cube_H
//#define BC_Cube_H
//
//#include "BC_Tensor_Base/Tensor_Base.h"
//
//namespace BC {
//
//template<class T, class Mathlib>
//class Cube : public Tensor_Base<Cube<T, Mathlib>> {
//
//	using parent_class = Tensor_Base<Cube<T, Mathlib>>;
//
//public:
//
//	__BCinline__ static constexpr int DIMS() { return 3; }
//	using parent_class::operator=;
//	using parent_class::operator[];
//	using parent_class::operator();
//
//	//constructors---------------------------------------------------
//
//	Cube(const Cube&  v) : parent_class(v) {}
//	Cube(	   Cube&& v) : parent_class(v) {}
//
//	explicit Cube(int rows = 0, int cols = 1, int pages = 1) : parent_class(Shape<3>(rows, cols, pages)) {}
//	explicit Cube(Shape<DIMS()> shape) : parent_class(shape)  {}
//
//	template<class... params>
//	Cube(const params&... p) : parent_class(p...) {}
//
//	//copy_operators--------------------------------------------------
//
//	Cube& operator =(const Cube& t)  { return parent_class::operator=(t); }
//	Cube& operator =(	   Cube&& t) { return parent_class::operator=(t); }
//
//	template<class U>
//	Cube& operator = (const Cube<U, Mathlib>& t) { return parent_class::operator=(t); }
//
//};
//}
//#endif /* Cube_H */
