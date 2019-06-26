
#ifndef BLACKCAT_NEURALNETWORK_UNARYFUNCTION_H_
#define BLACKCAT_NEURALNETWORK_UNARYFUNCTION_H_

namespace BC {
namespace nn {
namespace detail {

template<class T>
using query_cached_dx =	decltype(std::declval<T>().cached_dx);

template<class T, class voider=void>
struct detect_cached_dx : std::false_type {};

template<class T>
struct detect_cached_dx<T,
		std::enable_if_t<
				BC::meta::true_v<
					decltype(std::declval<T>().cached_dx)>
			>
		> : std::true_type {};


}

template<class SystemTag, class ValueType, class Functor>
class Function : public Layer_Base {

public:

	Functor function;

	using system_tag = SystemTag;
	using value_type = ValueType;

	using mat = BC::Matrix<ValueType, BC::Allocator<SystemTag, ValueType>>;
    using vec = BC::Vector<ValueType, BC::Allocator<SystemTag, ValueType>>;

    using mat_view = BC::Matrix_View<ValueType, BC::Allocator<SystemTag, ValueType>>;

private:
    mat y;
    mat_view x;

public:

    Function(int inputs, Functor function_=Functor()):
        Layer_Base(inputs, inputs),
        function(function_) {}

    template<class Matrix>
    const auto& forward_propagation(const Matrix& x_) {
    	x = mat_view(x_);
        return y = function(x);
    }
    template<class Matrix>
    auto back_propagation(const Matrix& dy) {
    	return BC::meta::constexpr_ternary<detail::detect_cached_dx<Functor>::value>(
    		BC::meta::bind([](auto function, auto& y, auto& dy){
        		return function.cached_dx(y) % dy;
    		}, function, y, dy),

    		BC::meta::bind([](auto function, auto& x, auto& dy){
   	        	return function.dx(x) % dy;
        	}, function, x, dy)
    	);
    }
    void update_weights() {}

    void set_batch_size(int x) {
        y = mat(this->numb_outputs(), x);
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
