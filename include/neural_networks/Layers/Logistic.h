///*
// * Logistic.h
// *
// *  Created on: Jun 2, 2019
// *      Author: joseph
// */
//
//#ifndef BLACKCAT_NEURALNETWORKS_LOGISTIC_H_
//#define BLACKCAT_NEURALNETWORKS_LOGISTIC_H_
//
//namespace BC {
//namespace nn {
//
//
//template<class SystemTag, class ValueType>
//class Logistic : public Layer_Base {
//
//public:
//
//	using system_tag = SystemTag;
//	using value_type = ValueType;
//
//	using mat = BC::Matrix<ValueType, BC::Allocator<SystemTag, ValueType>>;
//    using vec = BC::Vector<ValueType, BC::Allocator<SystemTag, ValueType>>;
//
//    using mat_view = BC::Matrix_View<ValueType, BC::Allocator<SystemTag, ValueType>>;
//
//private:
//
//    ValueType lr = 0.03;
//
//    mat y;           //outputs
//    mat_view x;      //inputs
//
//public:
//
//    Logistic(int inputs):
//        Layer_Base(inputs, inputs) {}
//
//    template<class Matrix>
//    const auto& forward_propagation(const Matrix& x) {
//        return y = BC::logistic(x);
//    }
//    template<class Matrix>
//    auto back_propagation(const Matrix& dy) {
//    	return dy; //faster than actually calculating the derivative
//    				//Todo change back to cached_dx_logistic when output error layer is changed to msse.
////        return BC::cached_dx_logistic(y) % dy;
//    }
//    void update_weights() {}
//
//    void set_batch_size(int x) {
//        y = mat(this->numb_outputs(), x);
//    }
//};
//
//template<class ValueType, class SystemTag>
//Logistic<SystemTag, ValueType> logistic(SystemTag system_tag, int inputs) {
//	return Logistic<SystemTag, ValueType>(inputs);
//}
//template<class SystemTag>
//auto logistic(SystemTag system_tag, int inputs) {
//	return Logistic<SystemTag, typename SystemTag::default_floating_point_type>(inputs);
//}
//
//}
//}
//
//
//
//#endif /* LOGISTIC_H_ */
