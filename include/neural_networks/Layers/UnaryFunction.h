
#ifndef BLACKCAT_NEURALNETWORK_UNARYFUNCTION_H_
#define BLACKCAT_NEURALNETWORK_UNARYFUNCTION_H_

namespace BC {
namespace nn {

template<class SystemTag, class ValueType, class Functor>
struct Function : public Layer_Base {

	Functor function;

	using system_tag = SystemTag;
	using value_type = ValueType;

	using mat = BC::Matrix<ValueType, BC::Allocator<SystemTag, ValueType>>;

public:

    Function(int inputs, Functor function_=Functor()):
        Layer_Base(inputs, inputs),
        function(function_) {}

    template<class Matrix>
    auto forward_propagation(const Matrix& x) {
        return function(x);
    }
    template<class X, class Delta>
    auto back_propagation(const X& x, const Delta& dy) {
    	return function.dx(x) % dy;
    }

    void update_weights() {}
    void set_batch_size(int x) {}
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
