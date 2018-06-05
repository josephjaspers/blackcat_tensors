///*
// * BLAS_Evaluator.h
// *
// *  Created on: Jun 5, 2018
// *      Author: joseph
// */
//
//#ifndef BLAS_EVALUATOR_H_
//#define BLAS_EVALUATOR_H_
//
//
//namespace BC {
//namespace Base {
//
//template<bool BARRIER = true, class derived_t>
//static void evaluate(const Tensor_Operations<derived_t>& tensor) {
//	using mathlib_type = _mathlib<derived_t>;
//
//	tensor.as_derived().internal().eval();
//
//	static constexpr int iterator_dimension = _functor<derived_t>::ITERATOR();
//	if (BARRIER)
//		mathlib_type::template dimension<iterator_dimension>::eval(tensor.as_derived().internal());
//	else
//		mathlib_type::template dimension<iterator_dimension>::eval_unsafe(tensor.as_derived().internal());
//}
//
//}
//}
//
//
//
//#endif /* BLAS_EVALUATOR_H_ */
