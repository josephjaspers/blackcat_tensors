///*
// * Core_Transpose.h
// *
// *  Created on: Jun 9, 2018
// *      Author: joseph
// */
//
//#ifndef CORE_TRANSPOSE_H_
//#define CORE_TRANSPOSE_H_
//
//#include "Core_Base.h"
//
//
//namespace BC{
//namespace internal {
//
//	template<class PARENT>
//	struct Tensor_Transpose : Tensor_Core_Base<Tensor_Transpose<PARENT>, PARENT::DIMS()> {
//
//		using scalar_type = _scalar<PARENT>;
//
//		__BCinline__ static constexpr int DIMS() 	 { return PARENT::DIMS(); }
//		__BCinline__ static constexpr int ITERATOR() { return DIMS(); }
//
//		static_assert(DIMS() <= 2, "TENSOR_TRANSPOSITION ONLY DEFINED FOR MATRICES AND VECTORS");
//
//		PARENT parent;
//		scalar_type* array;
//
//		Tensor_Transpose(const scalar_type* array_, PARENT parent_) :
//			array (const_cast<scalar_type*>(array_)), parent(parent_) {}
//
//
//		__BCinline__ int rows() const { return parent.cols(); }
//		__BCinline__ int cols() const { return parent.rows(); }
//		__BCinline__ const auto inner_shape() const { return l_array([=](int i) { return i == 0 ? parent.cols() : i == 1 ? parent.rows() : 1; }); }
//		__BCinline__ const auto outer_shape() const { return parent.outer_shape(); }
//		__BCinline__ std::enable_if_t<DIMS() == 1, const scalar_type&> operator () (int m) const { return array[m]; }
//		__BCinline__ std::enable_if_t<DIMS() == 1, 		 scalar_type&> operator () (int m) 		 { return array[m]; }
//
//		__BCinline__ std::enable_if_t<DIMS() == 2, const scalar_type&> operator () (int m, int n) const { return parent(n, m); }
//		__BCinline__ std::enable_if_t<DIMS() == 2, 		 scalar_type&> operator () (int m, int n) { return parent(n, m); }
//
//		__BCinline__ auto getIterator() const -> decltype(parent.getIterator()) { return parent.getIterator(); }
//		__BCinline__ auto getIterator() 	  -> decltype(parent.getIterator()) { return parent.getIterator(); }
//
//	};
//
//}
//}
//
//
//
//#endif /* CORE_TRANSPOSE_H_ */
