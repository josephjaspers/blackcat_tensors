
#ifndef BLACKCAT_NEURALNETWORK_UNARYFUNCTION_H_
#define BLACKCAT_NEURALNETWORK_UNARYFUNCTION_H_

#include <string>

namespace BC {
namespace nn {

template<
	class SystemTag,
	class ValueType,
	class Functor,
	class InputDimension = BC::traits::Integer<1>>
struct Function:
		Layer_Base<Function<SystemTag, ValueType, Functor, InputDimension>>
{
	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;
	using self_type = Function<SystemTag, ValueType, Functor,InputDimension>;
	using parent_type = Layer_Base<self_type>;
	using input_tensor_dimension = InputDimension;
	using output_tensor_dimension = InputDimension;

	using shape_type = BC::Dim<input_tensor_dimension::value>;

	Functor function;
	shape_type m_input_shape;

	Function(shape_type inputs, Functor function_=Functor()):
		parent_type(bc_get_classname_of(function), inputs.size(), inputs.size()),
		function(function_),
		m_input_shape (inputs) {}

	shape_type get_input_shape() const {
		return m_input_shape;
	}

	shape_type get_output_shape() const {
		return m_input_shape;
	}

	template<class Matrix>
	auto forward_propagation(const Matrix& x) {
		return function(x);
	}

	template<class X, class Delta>
	auto back_propagation(const X& x, const Delta& dy) {
		return function.dx(x) % dy;
	}
};


template<class ValueType, class SystemTag, class Functor>
Function<SystemTag, ValueType, Functor> function(SystemTag system_tag, int inputs, Functor function=Functor()) {
	return Function<SystemTag, ValueType, Functor>(BC::Dim<1>{inputs}, function);
}

template<class SystemTag, class Functor>
auto function(SystemTag system_tag, int inputs, Functor function=Functor()) {
	return Function<SystemTag, typename SystemTag::default_floating_point_type, Functor>(BC::Dim<1>{inputs}, function);
}


template<class ValueType, class SystemTag, class Functor, int X>
Function<SystemTag, ValueType, Functor> function(SystemTag system_tag, Dim<X> shape, Functor function=Functor()) {
	return Function<SystemTag, ValueType, Functor, BC::traits::Integer<X>>(shape, function);
}

template<class SystemTag, class Functor, int X>
auto function(SystemTag system_tag, BC::Dim<X> shape, Functor function=Functor()) {
	return Function<SystemTag, typename SystemTag::default_floating_point_type, Functor, BC::traits::Integer<X>>(shape, function);
}


}
}


#endif 
