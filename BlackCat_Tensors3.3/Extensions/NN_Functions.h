#ifndef BLACKCAT_TENSOR_FUNCTIONS
#define BLACKCAT_TENSOR_FUNCTIONS

#include <cmath>

namespace BC {

namespace NN_Functions {

struct Sigmoid {

	template<class T> __BCinline__
	T operator () (T t) const {
		static constexpr T e = 2.71828;

		return 1 / (1 + pow(e, - t));
	}
};

struct CachedSigmoidDeriv {

	template<class T> __BCinline__
	T operator () (T t) const {
		return t * (1 - t);
	}
};

struct Tanh {

	template<class T> __BCinline__
	T operator () (T t) const {
		static constexpr double e = 2.71828;

		return (pow(e, t) - pow(e, -t)) /
		(pow(e, t) + pow(e, -t));
	}
};

struct CachedTanhDeriv {
	template<class T> __BCinline__
	 T operator () (T t) const {
		return 1 - powf(t, 2);
	}
};
template<template<class, class > class tensor, class T, class ml>
auto sigmoid(const tensor<T, ml> & x) {
	return x.un_expr(Sigmoid());
}

template<template<class, class > class tensor, class T, class ml>
auto sigmoidDeriv(const tensor<T, ml> & x) {
	return x.un_expr(CachedSigmoidDeriv());
}

template<template<class, class > class tensor, class T, class ml>
auto tanh(const tensor<T, ml> & x) {
	return x.un_expr(Tanh());
}
template<template<class, class > class tensor, class T, class ml>
auto tanhDeriv(const tensor<T, ml> & x) {
	return x.un_expr(CachedTanhDeriv());
}

}

namespace NN_Abreviated_Functions {
using namespace NN_Functions;
template<template<class, class > class tensor, class T, class ml>
auto g(const tensor<T, ml>& x) {
	return x.un_expr(Sigmoid());
}

template<template<class, class > class tensor, class T, class ml>
auto gd(const tensor<T, ml>& x) {
	return x.un_expr(CachedSigmoidDeriv());
}

template<template<class, class > class tensor, class T, class ml>
auto h(const tensor<T, ml>& x) {
	return x.un_expr(Tanh());
}

template<template<class, class > class tensor, class T, class ml>
auto hd(const tensor<T, ml>& x) {
	return x.un_expr(CachedTanhDeriv());
}


}

}
#endif
