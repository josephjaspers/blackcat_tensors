///*
// * old_code.h
// *
// *  Created on: Nov 27, 2017
// *      Author: joseph
// */
//
//#ifndef TMP_STORAGE_NOTNEEDEDCURRENTLY_OLD_CODE_H_
//#define TMP_STORAGE_NOTNEEDEDCURRENTLY_OLD_CODE_H_
//
//
//
////
////template<class oper, class ml, class l, class r, int ... dimensions>
////struct Tensor_Ace<binary_expression<oper, ml, l, r, dimensions...>, binary_expression, dimensions...> : public Shape<dimensions...> {
////
////	using functor_type = binary_expression<oper, ml, l, r, dimensions...>;
////
////	auto data() {
////		return static_cast<functor_type&>(*this);
////	}
////	auto data() const {
////		return static_cast<const functor_type&>(*this);
////	}
////};
////
////template<class oper, class T, class ml, int ... dimensions>
////struct Tensor_Ace<unary_expression<oper, T, ml, dimensions...>, binary_expression, dimensions...> : public Shape<dimensions...> {
////
////	using functor_type = unary_expression<oper, T, ml, dimensions...>;
////
////	auto data() {
////		return static_cast<functor_type&>(*this);
////	}
////	auto data() const {
////		return static_cast<const functor_type&>(*this);
////	}
////};
////
////template<class T, class ml, int r, int c>
////struct Tensor_Ace<transpose_expression<T, ml, r, c>, binary_expression, r, c> : public Shape<r, c> {
////
////	using functor_type = transpose_expression<T, ml,r, c>;
////
////	auto data() {
////		return static_cast<functor_type&>(*this);
////	}
////	auto data() const {
////		return static_cast<const functor_type&>(*this);
////	}
////};
////
//
//#endif /* TMP_STORAGE_NOTNEEDEDCURRENTLY_OLD_CODE_H_ */
