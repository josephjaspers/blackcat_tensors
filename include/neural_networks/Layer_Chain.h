
#ifndef BLACKCAT_TUPLE
#define BLACKCAT_TUPLE

#include "Layer_Manager.h"

namespace BC {
namespace nn {

/*
 * The LayerChain implements a 'bi-directional-tuple'.
 * This bi-directional tuple stores the Layers inside of it.
 *
 * It supports reverse and forward iteration which allows it to be good container for neural networks.
 * (Forward iteration for forward propagation, reverse for back-prop)
 */

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
    using LayerType = std::conditional_t<index == 0, Input_Layer_Manager<CurrentLayer>, Layer_Manager<CurrentLayer>>;

    LayerType layer;

    LayerChain(CurrentLayer f, Layers... layers):
    	parent(layers...),
    	layer(f) {}

    const auto& head() const { return head_impl(BC::traits::truth_type<index==0>()); }
          auto& head()       { return head_impl(BC::traits::truth_type<index==0>()); }
    const auto& tail() const { return tail_impl(BC::traits::truth_type<sizeof...(Layers)==0>()); }
          auto& tail()       { return tail_impl(BC::traits::truth_type<sizeof...(Layers)==0>()); }

    const auto& next() const { return next_impl(BC::traits::truth_type<sizeof...(Layers) != 0>()); }
          auto& next()       { return next_impl(BC::traits::truth_type<sizeof...(Layers) != 0>()); }
    const auto& prev() const { return prev_impl(BC::traits::truth_type<index != 0>()); }
          auto& prev()       { return prev_impl(BC::traits::truth_type<index != 0>()); }

	template<class T> auto fp(const T& tensor) {
		return fp_impl(tensor, BC::traits::truth_type<sizeof...(Layers) !=0>());
	}

	template<class T> const auto bp(const T& tensor) {
		return bp_impl(tensor, BC::traits::truth_type<index!=0>());
	}

    template<class function> void for_each(function f) {
    	for_each_impl(f, BC::traits::truth_type<sizeof...(Layers) != 0>());
    }

    template<class function> void for_each_node(function f) {
    	for_each_node_impl(f, BC::traits::truth_type<sizeof...(Layers) != 0>());
    }


    //------------------------ implementation --------------------------- //
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

    template<class Function> void for_each_node_impl(Function f, std::true_type)  { f(*this); this->next().for_each_node(f); }
    template<class Function> void for_each_node_impl(Function f, std::false_type) { f(*this); }

	template<class T> const auto fp_impl(const T& tensor, std::true_type) {
		return this->next().fp(layer.forward_propagation(tensor));
	}
	template<class T> const auto bp_impl(const T& tensor, std::true_type) {
		return this->prev().bp(layer.back_propagation(tensor));
	}
	template<class T> const auto fp_impl(const T& tensor, std::false_type) {
		return layer.forward_propagation(tensor);
	}
	template<class T> const auto bp_impl(const T& tensor, std::false_type) {
		return layer.back_propagation(tensor);
	}
};

}
}
#endif
