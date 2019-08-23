/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *	  Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYERS_LOGGING_LAYER_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYERS_LOGGING_LAYER_H_

#include "Output_Layer.h"

namespace BC {
namespace nn {

struct Mean_Absolute_Error {
	template<class T>
	auto operator () (const T& expression) const {
		return BC::sum(BC::abs(expression)) / expression.size();
	}
} MAE;

struct Root_Mean_Squared_Error {
	template<class T>
	auto operator () (const T& expression) const {
		return BC::sqrt(BC::sum(BC::pow2(expression)) / expression.size());
	}
} RMSE;

struct Mean_Squared_Error {
	template<class T>
	auto operator () (const T& expression) const {
		return BC::sum(BC::pow2(expression)) / expression.size();
	}
} MSE;

template<class SystemTag, class ValueType, class ErrorFunction=Mean_Absolute_Error>
struct Logging_Output_Layer : Output_Layer<SystemTag,ValueType> {

	using parent = Output_Layer<SystemTag,ValueType>;
	using system_tag = SystemTag;
	using value_type = ValueType;

	bool logging_enabled = true;
	unsigned curr_index = 0;
	unsigned skip_every_n_backprops = 10;

	ErrorFunction error_function;
	std::ostream& logger;

	Logging_Output_Layer(std::ostream& logger_, BC::size_t inputs, ErrorFunction error_function_):
		parent(inputs),
		error_function(error_function_),
		logger(logger_) {}

	template <class Tensor>
	const auto& forward_propagation(const Tensor& x) {
		return x;
	}

	Logging_Output_Layer& skip_every(unsigned skip_every_n_backprops_) {
		skip_every_n_backprops = skip_every_n_backprops_;
		return *this;
	}

	Logging_Output_Layer& enable_logging(bool enable_logging=true) {
		logging_enabled = enable_logging;
		return *this;
	}

	template <class TensorX, class TensorY>
	auto back_propagation(const TensorX& x, const TensorY& y) {
		curr_index++;

		if (logging_enabled && curr_index % skip_every_n_backprops == 0)
			logger << "Batch index: " << curr_index << " loss: " << error_function(x-y) << "\n";
		return x - y;
	}
};

#ifndef BC_CLING_JIT
template<
	class ValueType,
	class SystemTag,
	class ErrorFunction=Mean_Absolute_Error>
Logging_Output_Layer<SystemTag, ValueType> logging_output_layer(
		SystemTag system_tag,
		BC::size_t inputs,
		ErrorFunction error_function=ErrorFunction(), std::ostream& os=std::cout) {
	return Logging_Output_Layer<SystemTag, ValueType>(os, inputs, error_function);
}
template<
	class SystemTag,
	class ErrorFunction=Mean_Absolute_Error>
auto logging_output_layer(
		SystemTag system_tag,
		BC::size_t inputs,
		ErrorFunction
		error_function=ErrorFunction(), std::ostream& os=std::cout) {

	return Logging_Output_Layer<
			SystemTag,
			typename SystemTag::default_floating_point_type,
			ErrorFunction>(os, inputs, error_function);
}
#endif

template<class ErrorFunction=Mean_Absolute_Error>
auto logging_output_layer(int inputs, ErrorFunction error_function=ErrorFunction(), std::ostream& os=std::cout) {
	return Logging_Output_Layer<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type, ErrorFunction>(os, inputs, error_function);
}


}
}



#endif /* FEEDFORWARD_CU_ */
