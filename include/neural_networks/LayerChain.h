
#ifndef BLACKCAT_TUPLE
#define BLACKCAT_TUPLE

#include "LayerCache.h"

namespace BC {
namespace nn {


template<int index, class Derived, class...>
struct LayerChain {};

template<int index, class Derived,class CurrentLayer, class... Layers>
struct LayerChain<index, Derived, CurrentLayer, Layers...>
: LayerChain<index + 1, LayerChain<index, Derived, CurrentLayer, Layers...>, Layers...> {

    using self   = LayerChain<index, Derived, CurrentLayer, Layers...>;
    using parent = LayerChain<index + 1, self, Layers...>;
    using type   = CurrentLayer;

    using value_type = typename CurrentLayer::value_type;
    using system_tag = typename CurrentLayer::system_tag;
    static constexpr int tensor_dimension = 1;//CurrentLayer::tensor_dimension;

    LayerCache<tensor_dimension, value_type, system_tag> m_cacher;
    CurrentLayer layer;

    LayerChain(CurrentLayer f, Layers... layers):
    	parent(layers...),
    	layer(f) {}

private:
    template<int ADL=0> const auto& head_impl(std::true_type) const { return *this; }
    template<int ADL=0>       auto& head_impl(std::true_type)       { return *this; }
    template<int ADL=0> const auto& head_impl(std::false_type) const { return prev().head(); }
    template<int ADL=0>       auto& head_impl(std::false_type)       { return prev().head(); }

    template<int ADL=0> const auto& tail_impl(std::true_type) const { return *this; }
    template<int ADL=0>       auto& tail_impl(std::true_type)       { return *this; }
    template<int ADL=0> const auto& tail_impl(std::false_type) const { return next().tail(); }
    template<int ADL=0>       auto& tail_impl(std::false_type)       { return next().tail(); }

    template<int ADL=0> const auto& next_impl(std::true_type) const { return static_cast<const parent&>(*this); }
	template<int ADL=0>       auto& next_impl(std::true_type)       { return static_cast<parent&>(*this); }
	template<int ADL=0> const auto& next_impl(std::false_type) const { return *this; }
	template<int ADL=0>       auto& next_impl(std::false_type)       { return *this; }

    template<int ADL=0> const auto& prev_impl(std::true_type) const { return static_cast<const Derived&>(*this); }
    template<int ADL=0>       auto& prev_impl(std::true_type)       { return static_cast<Derived&>(*this); }
    template<int ADL=0> const auto& prev_impl(std::false_type) const { return *this; }
    template<int ADL=0>       auto& prev_impl(std::false_type)       { return *this; }

    template<class Function> void for_each_impl(Function f, std::true_type)  { f(layer); this->next().for_each(f); }
    template<class Function> void for_each_impl(Function f, std::false_type) { f(layer); }

    template<class T> const auto& call_forward(const T& tensor, std::true_type requires_outputs) {
    	static constexpr bool is_batched = T::tensor_dimension != tensor_dimension;
    	auto cache_tensor_dimension = BC::traits::Integer<is_batched + parent::tensor_dimension>();
		return layer.forward_propagation(tensor, this->next().m_cacher.get_last(cache_tensor_dimension));
    }
    template<class T> auto call_forward(const T& tensor, std::false_type requires_outputs)
    -> decltype(layer.forward_propagation(tensor)) {
		return layer.forward_propagation(tensor);
    }


	template<class T> const auto& fp_impl(const T& tensor, std::true_type) {
		return this->next().fp(call_forward(tensor, typename layer_traits<CurrentLayer>::forward_requires_outputs()));
	}
	template<class T> const auto bp_impl(const T& tensor, std::true_type) {
		return this->prev().bp(
				layer.back_propagation(
						m_cacher.get_last(BC::traits::Integer<T::tensor_dimension>()), tensor));
	}
	template<class T> const auto& fp_impl(const T& tensor, std::false_type) {
		return call_forward(tensor, typename layer_traits<CurrentLayer>::forward_requires_outputs());
	}
	template<class T> const auto bp_impl(const T& tensor, std::false_type) {
		return layer.back_propagation(m_cacher.m_batched_cache.back(), tensor);
	}


public:

    const auto& head() const { return head_impl(BC::traits::truth_type<index==0>()); }
          auto& head()       { return head_impl(BC::traits::truth_type<index==0>()); }
    const auto& tail() const { return tail_impl(BC::traits::truth_type<sizeof...(Layers)==0>()); }
          auto& tail()       { return tail_impl(BC::traits::truth_type<sizeof...(Layers)==0>()); }

    const auto& next() const { return next_impl(BC::traits::truth_type<sizeof...(Layers) != 0>()); }
          auto& next()       { return next_impl(BC::traits::truth_type<sizeof...(Layers) != 0>()); }
    const auto& prev() const { return prev_impl(BC::traits::truth_type<index != 0>()); }
          auto& prev()       { return prev_impl(BC::traits::truth_type<index != 0>()); }

	template<class T> const auto& fp(const T& tensor) {
		m_cacher.cache(tensor);
		return fp_impl(m_cacher.m_batched_cache.back(), BC::traits::truth_type<sizeof...(Layers)!=0>());
	}
	template<class T> const auto bp(const T& tensor) {
		return bp_impl(tensor, BC::traits::truth_type<index!=0>());
	}

    template<class function> void for_each(function f) {
    	for_each_impl(f, BC::traits::truth_type<sizeof...(Layers) != 0>());
    }
};

}
}
#endif
