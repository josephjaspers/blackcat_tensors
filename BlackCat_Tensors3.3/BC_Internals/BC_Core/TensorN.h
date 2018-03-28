///*
// * Cube.h
// *
// *  Created on: Dec 30, 2017
// *      Author: joseph
// */
//
//#ifndef BC_TENSOR_N
//#define BC_TENSOR_N
//x
//#include "TensorBase.h"
//
//
//namespace BC {
//
//
//template<class T, class Mathlib>
//class Tensor : public TensorBase<Tensor<T, Mathlib>> {
//
//	using parent_class = TensorBase<Tensor<T, Mathlib>>;
//
//public:
//	static constexpr int DIMS() { return 1; }
//
//
//	using scalar = T;
//	using parent_class::operator=;
//	using parent_class::operator[];
//	using parent_class::operator();
//
//
//	Tensor(const Tensor&  v) : parent_class(v) {}
//	Tensor(	  Tensor&& v) : parent_class(v) {}
//	Tensor(const Tensor&& v) : parent_class(v) {}
//
//	explicit Tensor(std::vector<int> ints) : parent_class(ints) {}
//
//	template<class U> 		  Tensor(const Tensor<U, Mathlib>&  t) : parent_class(t) {}
//	template<class U> 		  Tensor(	     Tensor<U, Mathlib>&& t) : parent_class(t) {}
//	template<class... params> Tensor(const params&... p) : parent_class(p...) {}
//
//	Tensor& operator =(const Tensor& t)  { return parent_class::operator=(t); }
//	Tensor& operator =(const Tensor&& t) { return parent_class::operator=(t); }
//	Tensor& operator =(	  Tensor&& t) { return parent_class::operator=(t); }
//	template<class U>
//	Tensor& operator = (const Tensor<U, Mathlib>& t) { return parent_class::operator=(t); }
//};
//
////template<class T, class ml = CPU, class... integers>
////auto Tensor(integers... ints) {
////	return typename D<sizeof...(ints)>::template Tensor<T, ml>(std::vector<int> {ints...});
////}
////
//
//}//End Namespace BC
//
//#endif /* Cube_H */
