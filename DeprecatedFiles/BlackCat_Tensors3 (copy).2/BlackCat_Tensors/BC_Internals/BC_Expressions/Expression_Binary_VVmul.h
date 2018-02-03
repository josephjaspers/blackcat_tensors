///*
// * Expression_Binary_VVmul.h
// *
// *  Created on: Dec 20, 2017
// *      Author: joseph
// */
//
//#ifndef EXPRESSION_BINARY_VVMUL_H_
//#define EXPRESSION_BINARY_VVMUL_H_
//#include <cmath>
//#include "../BlackCat_Internal_GlobalUnifier.h"
//
//namespace BC {
//
//template<
//	class T,
//	class COL_VECTOR,
//	class ROW_VECTOR>
//struct binary_expression_VVmul_outer : expression<T, binary_expression_VVmul_outer<T, COL_VECTOR, ROW_VECTOR>>{
//
//	/*
//	 * Outer product (always between Col Vector and Row Vector
//	 */
//
//	const COL_VECTOR& vecL;
//	const ROW_VECTOR& vecR;
//
//	const typename COL_VECTOR::functor_type& lv = vecL.data();
//	const typename ROW_VECTOR::functor_type& rv = vecR.data();
//
//	binary_expression_VVmul_outer(const COL_VECTOR& lv, const ROW_VECTOR& rv) : vecL(lv), vecR(rv) {}
//	binary_expression_VVmul_outer(const binary_expression_VVmul_outer&) = default;
//	~binary_expression_VVmul_outer() {};
//	T operator [] (int index) const {
//		return lv[(int)std::ceil(index / vecR.cols())] * rv[(int)index % vecR.cols()];
//	}
//};
//
//}
//
//#endif /* EXPRESSION_BINARY_VVMUL_H_ */
