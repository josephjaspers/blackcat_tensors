#ifndef BC_NEURAL_NETWORK
#define BC_NEURAL_NETWORK

#include <tuple>

template<class derived, class... layers>
class LayerManager : LayerManager<LayerManager<derived, layers...>, layers...> {

	using prev_t = derived;
	using next_t = LayerManager<derived, layers...>; //self
	using layer_t = void;

	auto& next() {
		return *this;
	}
	auto& prev() {
		return static_cast<prev_t*>(*this);
	}

	template<class tensor>
	auto backpropagation(const tensor& x) {
		 prev().backpropagation(x);
	}
	template<class tensor>
	auto forwardpropagation(const tensor& x) {
		return x;
	}
};


template<class derived, class layer_t,  class... layers>
class LayerManager<derived, layer_t, layers...> : LayerManager<LayerManager<derived, layer_t, layers...>, layers...> {

	using self_t = LayerManager<derived, layer, layers...>;
	using prev_t = derived;
	using next_t = LayerManager<self_t, layers...>;

	layer_t layer;

	auto& next() {
		return static_cast<next_t&>(this);
	}
	auto& prev() {
		return static_cast<prev_t*>(*this);
	}

	template<class tensor>
	auto backpropagation(const tensor& x) {
		 prev().backpropagation(layer.backpropagation(x));
	}
	template<class tensor>
	auto forwardpropagation(const tensor& x) {
		return next().forwardpropagation(layer.forwardpropagation(x));
	}
};

#endif
