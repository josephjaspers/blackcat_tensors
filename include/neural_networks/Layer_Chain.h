
#ifndef BLACKCAT_TUPLE
#define BLACKCAT_TUPLE

#include "Layer_Manager.h"

namespace BC {
namespace nn {
namespace detail {

template<class T>
using query_layer_type = typename T::layer_type;

template<
	class Neural_Network_Is_Recurrent,
	class Derived,
	class Layer>
using LayerManager = Layer_Manager<Derived, Layer, Neural_Network_Is_Recurrent>;

}

/**
 * The LayerChain implements a 'bi-directional-tuple'.
 * This bi-directional tuple stores the Layers inside of it.
 *
 * It supports reverse and forward iteration which allows it to be good container for neural networks.
 * (Forward iteration for forward propagation, reverse for back-prop)
 *
 *
 * Neural_Network_Is_Recurrent --> expects std::true_type || std::is_false_type
 */
template<class Index, class Neural_Network_Is_Recurrent, class Derived, class...>
struct LayerChain {};

template<
	class Index,
	class Neural_Network_Is_Recurrent,
	class Derived,
	class CurrentLayer,
	class... Layers>
struct LayerChain<
	Index,
	Neural_Network_Is_Recurrent,
	Derived,
	CurrentLayer,
	Layers...>:

	LayerChain<
		BC::traits::Integer<Index::value + 1>,
		Neural_Network_Is_Recurrent,
		LayerChain<Index, Neural_Network_Is_Recurrent, Derived, CurrentLayer, Layers...>,
		Layers...>,
	detail::LayerManager<
		Neural_Network_Is_Recurrent,
		LayerChain<Index, Neural_Network_Is_Recurrent, Derived, CurrentLayer, Layers...>,
		CurrentLayer> {	 //if not first layer

	using self_type = LayerChain<
		Index,
		Neural_Network_Is_Recurrent,
		Derived,
		CurrentLayer,
		Layers...>;

	using parent_type = LayerChain<
			BC::traits::Integer<Index::value + 1>,
			Neural_Network_Is_Recurrent,
			self_type,
			Layers...>;

	using layer_type = detail::LayerManager<
			Neural_Network_Is_Recurrent,
			self_type,
			CurrentLayer>;

	using next_layer_type = BC::traits::conditional_detected<
			detail::query_layer_type, parent_type, void>;

	using type   = CurrentLayer;
	using value_type = typename CurrentLayer::value_type;
	using system_tag = typename CurrentLayer::system_tag;

	using is_input_layer	  = BC::traits::truth_type<Index::value==0>;
	using is_output_layer	  = BC::traits::truth_type<sizeof...(Layers)==0>;
	using is_not_input_layer  = BC::traits::not_type<is_input_layer::value>;
	using is_not_output_layer = BC::traits::not_type<is_output_layer::value>;

	LayerChain(CurrentLayer f, Layers... layers):
		parent_type(layers...),
		layer_type(f) {}

	const auto& layer() const { return static_cast<const layer_type&>(*this); }
	auto& layer() { return static_cast<layer_type&>(*this); }

	const auto& get(BC::traits::Integer<0>) const { return layer(); }
		  auto& get(BC::traits::Integer<0>)	   { return layer(); }

	template<int X>
	const auto& get(BC::traits::Integer<X>) const { return next().get(BC::traits::Integer<X-1>()); }

	template<int X>
	auto& get(BC::traits::Integer<X>)	   { return next().get(BC::traits::Integer<X-1>()); }

	const auto& head() const { return head_impl(is_input_layer()); }
		  auto& head()	   { return head_impl(is_input_layer()); }
	const auto& tail() const { return tail_impl(is_output_layer()); }
		  auto& tail()	   { return tail_impl(is_output_layer()); }

	const auto& next() const { return next_impl(is_not_output_layer()); }
		  auto& next()	   { return next_impl(is_not_output_layer()); }
	const auto& prev() const { return prev_impl(is_not_input_layer()); }
		  auto& prev()	   { return prev_impl(is_not_input_layer()); }

	template<class T> auto fp(const T& tensor) {
		return fp_impl(tensor, is_not_output_layer());
	}

	template<class T> const auto bp(const T& tensor) {
		return bp_impl(tensor, is_not_input_layer());
	}

	template<class function> void for_each(function f) {
		for_each_impl(f, is_not_output_layer());
	}

	template<class function> void for_each_node(function f) {
		for_each_node_impl(f, is_not_output_layer());
	}

	template<class function> void for_each(function f) const {
		for_each_impl(f, is_not_output_layer());
	}

	template<class function> void for_each_node(function f) const {
		for_each_node_impl(f, is_not_output_layer());
	}


	//------------------------ implementation --------------------------- //
private:

	template<int ADL=0> const auto& head_impl(std::true_type) const { return *this; }
	template<int ADL=0>	   auto& head_impl(std::true_type)	   { return *this; }
	template<int ADL=0> const auto& head_impl(std::false_type) const { return prev().head(); }
	template<int ADL=0>	   auto& head_impl(std::false_type)	   { return prev().head(); }

	template<int ADL=0> const auto& tail_impl(std::true_type) const { return *this; }
	template<int ADL=0>	   auto& tail_impl(std::true_type)	   { return *this; }
	template<int ADL=0> const auto& tail_impl(std::false_type) const { return next().tail(); }
	template<int ADL=0>	   auto& tail_impl(std::false_type)	   { return next().tail(); }

	template<int ADL=0> const auto& next_impl(std::true_type) const { return static_cast<const parent_type&>(*this); }
	template<int ADL=0>	   auto& next_impl(std::true_type)	   { return static_cast<parent_type&>(*this); }
	template<int ADL=0> const auto& next_impl(std::false_type) const { return *this; }
	template<int ADL=0>	   auto& next_impl(std::false_type)	   { return *this; }

	template<int ADL=0> const auto& prev_impl(std::true_type) const { return static_cast<const Derived&>(*this); }
	template<int ADL=0>	   auto& prev_impl(std::true_type)	   { return static_cast<Derived&>(*this); }
	template<int ADL=0> const auto& prev_impl(std::false_type) const { return *this; }
	template<int ADL=0>	   auto& prev_impl(std::false_type)	   { return *this; }

	template<class Function> void for_each_impl(Function f, std::true_type) const {
		f(static_cast<const layer_type&>(*this)); this->next().for_each(f);
	}

	template<class Function> void for_each_impl(Function f, std::false_type) const {
		f(static_cast<const layer_type&>(*this));
	}

	template<class Function> void for_each_node_impl(Function f, std::true_type) const {
		f(*this); this->next().for_each_node(f);
	}

	template<class Function> void for_each_node_impl(Function f, std::false_type) const {
		f(*this);
	}

	template<class Function> void for_each_impl(Function f, std::true_type) {
		f(static_cast<layer_type&>(*this)); this->next().for_each(f);
	}

	template<class Function> void for_each_impl(Function f, std::false_type) {
		f(static_cast<layer_type&>(*this));
	}

	template<class Function> void for_each_node_impl(Function f, std::true_type)  {
		f(*this); this->next().for_each_node(f);
	}

	template<class Function> void for_each_node_impl(Function f, std::false_type) {
		f(*this);
	}


	template<class T> const auto fp_impl(const T& tensor, std::true_type has_next) {
		return this->next().fp(layer_type::forward_propagation(tensor));
	}

	template<class T> const auto fp_impl(const T& tensor, std::false_type has_next) {
		return layer_type::forward_propagation(tensor);
	}

	template<class T> const auto bp_impl(const T& tensor, std::true_type) {
		return this->prev().bp(layer_type::back_propagation(tensor));
	}

	template<class T> const auto bp_impl(const T& tensor, std::false_type) {
		return layer_type::back_propagation(tensor);
	}
};
}
}
#endif
