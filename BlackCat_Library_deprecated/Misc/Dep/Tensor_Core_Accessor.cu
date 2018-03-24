//
//
///*
// * Shape.h
// *
// *  Created on: Jan 18, 2018
// *      Author: joseph
// */
//
//#ifndef TENSOR_CORE_ACCESSOR_H
//#define TENSOR_CORE_ACCESSOR_H
//
//#include "Tensor_Core_Essentials.cu"
//namespace BC {
//template<class T, class Mathlib, class ranker> struct Tensor_Core;
//
//template<class T, class Mathlib, int inner, int outer>
//struct Tensor_Core<T, Mathlib, Rank<inner - 1, outer>> {
//	using core = Tensor_Core<T, Mathlib, Rank<inner, outer>>;
//	using accessor = Tensor_Core<T, Mathlib, Rank<inner - 2, outer>>;
//	template<int x> using sub_tensor = Tensor_Core<T, Mathlib, Rank<inner - x, outer>>;
//
//	static constexpr int RANK = inner - 1;
//	static constexpr int LD_RANK = outer;
//	static constexpr bool tensor_ownership = false;
//	static constexpr int BACK = RANK  - 1;
//
//	using scalar = typename MTF::determine_scalar<T>::type;
//	scalar* array;
//	int* is;
//	int* os;
//
//		  scalar* ary() { return array; }
//	const scalar* ary() const { return array; }
//
//	operator 	   scalar*()       { return array; }
//	operator const scalar*() const { return array; }
//
//	__BC_gcpu__	      scalar& operator [] (int index) 		{ return array[index]; };
//	__BC_gcpu__	const scalar& operator [] (int index) const { return array[index]; };
//
//	Tensor_Core(scalar* array, int* is, int* os) : array(array), is(is), os(os) {}
//
//	__BC_gcpu__ int dims() const { return RANK; }
//	__BC_gcpu__ int size() const { return os[BACK];    }
//	__BC_gcpu__ int rows() const { return RANK > 0 ? is[0] : 1; }
//	__BC_gcpu__ int cols() const { return RANK > 1 ? is[1] : 1; }
//	__BC_gcpu__ int dimension(int i) const { return RANK > i ? is[i] : 1; }
//	 void printDimensions() const { for (int i = 0; i < RANK; ++i) { std::cout << "["<< is[i] << "]"; } std::cout << std::endl; }
//	__BC_gcpu__ void printLDDimensions() const { for (int i = 0; i < RANK; ++i) { std::cout << "["<< os[i] << "]"; } std::cout << std::endl; }
//	__BC_gcpu__ int LD_rows() const { return RANK > 0 ? os[0] : 1; }
//	__BC_gcpu__ int LD_cols() const { return RANK > 1 ? os[1] : 1; }
//	__BC_gcpu__ int LDdimension(int i) const { return RANK > i + 1 ? os[i] : 1; }
//
//	__BC_gcpu__ const int* InnerShape() const { return is; }
//	__BC_gcpu__ const int* OuterShape() const { return os; }
//	void print() const { Mathlib::print(array, InnerShape(),dims(), 4); }
//
//	__BC_gcpu__ const scalar* data() const { return array; }
//	__BC_gcpu__ scalar* data()  		   { return array; }
//
//};
//}
//
//#endif /* SHAPE_H_ */
//
