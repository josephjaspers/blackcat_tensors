#ifndef BLACKCAT_TENSOR_FUNCTIONS
#define BLACKCAT_TENSOR_FUNCTIONS

#include <cmath>

namespace BC {

template<class> class Tensor_Base;

namespace NN_Functions {

struct Sigmoid {

	template<class T> __BCinline__
	T operator () (const T& t) const {
		static constexpr T e = 2.71828;

		return 1 / (1 + pow(e, - t));
	}
};

struct CachedSigmoidDeriv {

	template<class T> __BCinline__
	T operator () (const T& t) const {
		return t * (1 - t);
	}
};

struct Tanh {

	template<class T> __BCinline__
	T operator () (const T& t) const {
		static constexpr double e = 2.71828;

		return (pow(e, t) - pow(e, -t)) /
		(pow(e, t) + pow(e, -t));
	}
};

struct CachedTanhDeriv {
	template<class T> __BCinline__
	 T operator () (const T& t) const {
		return 1 - powf(t, 2);
	}
};
template<class T>
auto sigmoid(const Tensor_Base<T> & x) {
	return x.un_expr(Sigmoid());
}

template<class T>
auto sigmoidDeriv(const Tensor_Base<T> & x) {
	return x.un_expr(CachedSigmoidDeriv());
}

template<class T>
auto tanh(const Tensor_Base<T> & x) {
	return x.un_expr(Tanh());
}
template<class T>
auto tanhDeriv(const Tensor_Base<T> & x) {
	return x.un_expr(CachedTanhDeriv());
}

}

namespace NN_Abreviated_Functions {
using namespace NN_Functions;
template<class T>
auto g(const Tensor_Base<T>& x) {
	return x.un_expr(Sigmoid());
}

template<class T>
auto gd(const Tensor_Base<T>& x) {
	return x.un_expr(CachedSigmoidDeriv());
}

template<class T>
auto h(const Tensor_Base<T>& x) {
	return x.un_expr(Tanh());
}

template<class T>
auto hd(const Tensor_Base<T>& x) {
	return x.un_expr(CachedTanhDeriv());
}


}

}
#endif
