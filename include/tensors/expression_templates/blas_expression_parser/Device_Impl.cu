namespace BC {
namespace tensors {
namespace exprs { 
namespace blas_expression_parser {
namespace device_detail {

template<class T, class enabler = void>
struct get_value_impl {
	__device__
    static T& impl(T& scalar) {
        return scalar;
    }
	__device__
	static const T& impl(const T& scalar) {
		return scalar;
	}
};
template<class T>
struct get_value_impl<T*, void> {
	__device__
    static auto impl(const T* scalar) -> decltype(scalar[0]) {
		static constexpr T one = 1;
        return scalar == nullptr ? one : scalar[0];
    }
};
template<class T>
struct get_value_impl<T, std::enable_if_t<T::tensor_dimension==0>>  {
	__device__
	static auto impl(T scalar) -> decltype(scalar[0]) {
        return scalar[0];
    }
};

template<class T> __device__
static auto get_value(T scalar) {
	return get_value_impl<T>::impl(scalar);
}

template<class Scalar> __device__
static auto calculate_alpha_impl(Scalar value) {
	return get_value(value);
}
template<class ScalarFirst, class... Scalars>__device__
static auto calculate_alpha_impl(ScalarFirst value, Scalars... vals) {
	return get_value(value) * calculate_alpha_impl(vals...);
}

template<class ScalarOut, class... Scalars> __global__
static void calculate_alpha(ScalarOut output, Scalars... vals) {
	output[0] = calculate_alpha_impl(vals...);
}

}
}
}
}
}
