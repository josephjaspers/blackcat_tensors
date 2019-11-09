
#ifndef BLACKCAT_NEURALNETWORK_UNARYFUNCTION_H_
#define BLACKCAT_NEURALNETWORK_UNARYFUNCTION_H_

#include <string>

namespace BC {
namespace nn {

template<class SystemTag, class ValueType, class Functor>
struct Function:
		Layer_Base<Function<SystemTag, ValueType, Functor>> {

	Functor function;

	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;
	using parent_type = Layer_Base<Function<SystemTag, ValueType, Functor>>;

	Function(int inputs, Functor function_=Functor()):
		parent_type(bc_get_classname_of(function), inputs, inputs),
		function(function_) {}

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
	return Function<SystemTag, ValueType, Functor>(inputs, function);
}

template<class SystemTag, class Functor>
auto function(SystemTag system_tag, int inputs, Functor function=Functor()) {
	return Function<SystemTag, typename SystemTag::default_floating_point_type, Functor>(inputs, function);
}


}
}


#endif 
