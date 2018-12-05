
namespace BC {
namespace blas {
namespace device_impl {

template<class T, class U, class V> __global__
static void scalar_mul(T* t, U* u, V* v) {
    *t = *u * *v;
}
template<class T, class U, class V> __global__
static void scalar_mul(T* t, U u, V* v) {
    *t = u * *v;
}
template<class T, class U, class V> __global__
static void scalar_mul(T* t, U* u, V v) {
    *t = *u * v;
}
template<class T, class U, class V> __global__
static void scalar_mul(T* t, U u, V v) {
    *t = u * v;
}
}
}
}
