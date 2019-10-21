#ifndef BLACKCAT_TUPLE
#define BLACKCAT_TUPLE

#include "Layer_Manager.h"

namespace BC {
namespace nn {
namespace detail {

template<class T>
using query_layer_type = typename T::layer_type;

}

/**
 * Layer_Chain is an iterator-like object that connects different types of
 * neural-network layers and defines convenient iterator-like methods.
 *
 * It is tightly coupled with Layer_Manager
 * Layer_Manager expects LayerChain to derive from itself.
 *
 * This enables Layer_Manager (which inherits from a neural-network layer)
 * to cast-itself to the LayerChain class and than access the other layers of the network.
 *
 * This enables Layer_Manager's to have acess to all other layer's of the neural network.
 *
 * The Layer_Chain simply acts as an iterator-like object.
 * The Layer_Manager handles memory and time_indexes (for recurrent layers)
 * for actual neural-network layers.
 *
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
	Layer_Manager<
			LayerChain<Index, Neural_Network_Is_Recurrent, Derived, CurrentLayer, Layers...>,
			CurrentLayer,
			Neural_Network_Is_Recurrent> {	 //if not first layer

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

	using layer_type = Layer_Manager<
			self_type,
			CurrentLayer,
			Neural_Network_Is_Recurrent>;

	using next_layer_type = BC::traits::conditional_detected<
			detail::query_layer_type, parent_type, void>;

	using type   = CurrentLayer;
	using value_type = typename CurrentLayer::value_type;
	using system_tag = typename CurrentLayer::system_tag;

	using is_input_layer      = BC::traits::truth_type<Index::value==0>;
	using is_output_layer     = BC::traits::truth_type<sizeof...(Layers)==0>;
	using is_not_input_layer  = BC::traits::not_type<is_input_layer::value>;
	using is_not_output_layer = BC::traits::not_type<is_output_layer::value>;

	LayerChain(CurrentLayer f, Layers... layers):
		parent_type(layers...),
		layer_type(f) {}

	const auto& layer() const { return static_cast<const layer_type&>(*this); }
	      auto& layer()       { return static_cast<      layer_type&>(*this); }

	const auto& get(BC::traits::Integer<0>) const { return layer(); }
	      auto& get(BC::traits::Integer<0>)       { return layer(); }

	template<int X>
	const auto& get(BC::traits::Integer<X>) const {
		return next().get(BC::traits::Integer<X-1>());
	}

	template<int X>
	auto& get(BC::traits::Integer<X>) {
		return next().get(BC::traits::Integer<X-1>());
	}

	const auto& head() const { return head_impl(is_input_layer()); }
		  auto& head()	   { return head_impl(is_input_layer()); }
	const auto& tail() const { return tail_impl(is_output_layer()); }
		  auto& tail()	   { return tail_impl(is_output_layer()); }

	const auto& next() const { return next_impl(is_not_output_layer()); }
		  auto& next()	   { return next_impl(is_not_output_layer()); }
	const auto& prev() const { return prev_impl(is_not_input_layer()); }
		  auto& prev()	   { return prev_impl(is_not_input_layer()); }

	template<class function> void for_each(function f) {
		for_each_impl(f, is_not_output_layer());
	}

	template<class function> void for_each(function f) const {
		for_each_impl(f, is_not_output_layer());
	}

	template<class function, class Arg>
	auto for_each_propagate(function f, Arg&& arg) {
		return for_each_propagate_impl(f, arg, is_not_output_layer());
	}

	template<class function, class Arg>
	auto for_each_propagate(function f, Arg&& arg) const {
		return for_each_propagate_impl(f, arg, is_not_output_layer());
	}

	template<class function, class Arg>
	auto reverse_for_each_propagate(function f, Arg&& arg) {
		return reverse_for_each_propagate_impl(f, arg, is_not_input_layer());
	}

	template<class function, class Arg>
	auto reverse_for_each_propagate(function f, Arg&& arg) const {
		return reverse_for_each_propagate_impl(f, arg, is_not_input_layer());
	}

	//------------------------ implementation --------------------------- //
private:

	template<int ADL=0> const auto& head_impl(std::true_type ) const { return *this; }
	template<int ADL=0>       auto& head_impl(std::true_type )       { return *this; }
	template<int ADL=0> const auto& head_impl(std::false_type) const { return prev().head(); }
	template<int ADL=0>       auto& head_impl(std::false_type)       { return prev().head(); }

	template<int ADL=0> const auto& tail_impl(std::true_type ) const { return *this; }
	template<int ADL=0>       auto& tail_impl(std::true_type )       { return *this; }
	template<int ADL=0> const auto& tail_impl(std::false_type) const { return next().tail(); }
	template<int ADL=0>       auto& tail_impl(std::false_type)       { return next().tail(); }

	template<int ADL=0> const auto& next_impl(std::true_type ) const { return static_cast<const parent_type&>(*this); }
	template<int ADL=0>       auto& next_impl(std::true_type )       { return static_cast<parent_type&>(*this); }
	template<int ADL=0> const auto& next_impl(std::false_type) const { return *this; }
	template<int ADL=0>       auto& next_impl(std::false_type)       { return *this; }

	template<int ADL=0> const auto& prev_impl(std::true_type ) const { return static_cast<const Derived&>(*this); }
	template<int ADL=0>       auto& prev_impl(std::true_type )       { return static_cast<Derived&>(*this); }
	template<int ADL=0> const auto& prev_impl(std::false_type) const { return *this; }
	template<int ADL=0>       auto& prev_impl(std::false_type)       { return *this; }

	template<class Function>
	void for_each_impl(Function f, std::true_type) const {
		f(layer());
		this->next().for_each(f);
	}

	template<class Function>
	void for_each_impl(Function f, std::true_type) {
		f(layer()); this->next().for_each(f);
	}

	template<class Function>
	void for_each_impl(Function f, std::false_type) const {
		f(layer());
	}

	template<class Function>
	void for_each_impl(Function f, std::false_type) {
		f(layer());
	}

	template<class Function, class Arg>
	auto for_each_propagate_impl(Function f, Arg&& arg, std::true_type) const {
		return next().for_each_propagate(f, f(layer(), arg));
	}

	template<class Function, class Arg>
	auto for_each_propagate_impl(Function f, Arg&& arg, std::false_type) const {
		return f(layer(), arg);
	}

	template<class Function, class Arg>
	auto for_each_propagate_impl(Function f, Arg&& arg, std::true_type) {
		return next().for_each_propagate(f, f(layer(), arg));
	}

	template<class Function, class Arg>
	auto for_each_propagate_impl(Function f, Arg&& arg, std::false_type) {
		return f(layer(), arg);
	}

	template<class Function, class Arg>
	auto reverse_for_each_propagate_impl(Function f, Arg&& arg, std::true_type) const {
		return prev().reverse_for_each_propagate(f, f(layer(), arg));
	}

	template<class Function, class Arg>
	auto reverse_for_each_propagate_impl(Function f, Arg&& arg, std::true_type) {
		return prev().reverse_for_each_propagate(f, f(layer(), arg));
	}

	template<class Function, class Arg>
	auto reverse_for_each_propagate_impl(Function f, Arg&& arg, std::false_type) {
		return f(layer(), arg);
	}

	template<class Function, class Arg>
	auto reverse_for_each_propagate_impl(Function f, Arg&& arg, std::false_type) const {
		return f(layer(), arg);
	}

};
}
}
#endif
