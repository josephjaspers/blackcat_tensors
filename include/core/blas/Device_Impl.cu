
namespace BC {
namespace blas {
namespace device_impl {

template<class T, class enabler = void>
struct get_value_impl {
	__device__
    static auto impl(T scalar) {
        return scalar;
    }
};
template<class T>
struct get_value_impl<T, std::enable_if_t<!std::is_same<decltype(std::declval<T&>()[0]), void>::value>>  {
	__device__
	static auto impl(T scalar) {
        return scalar[0];
    }
};

template<class T> __device__
static auto get_value(T scalar) {
	return get_value_impl<T>::impl(scalar);
}

template<class Scalar> __device__
static auto scalar_mul_impl(Scalar value) {
	return get_value(value);
}
template<class ScalarFirst, class... Scalars>__device__
static auto scalar_mul_impl(ScalarFirst value, Scalars... vals) {
	return get_value(value) * scalar_mul_impl(vals...);
}

template<class ScalarOut, class... Scalars> __global__
static void scalar_mul(ScalarOut output, Scalars... vals) {
	output[0] = scalar_mul_impl(vals...);
}

}
}
}
